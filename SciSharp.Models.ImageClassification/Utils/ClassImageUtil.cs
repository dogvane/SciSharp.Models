using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Tensorflow;
using static Tensorflow.Binding;


namespace SciSharp.Models.ImageClassification
{
    /// <summary>
    /// 这个主要是用于放置一些静态方法
    /// 用于图片的预处理
    /// </summary>
    public static class ClassImageUtil
    {
        public static IDatasetV2 LoadImagesFromDirectory(string folder)
        {
            var folderLabels = GetLabelFromFolderName(folder);
            var labelConfigFile = Path.Combine(folder, "label.csv");
            var currentLabels = GetLabelFromConfigFile(labelConfigFile);

            // 合并已有的标签和现在目录发现的标签
            // 实际标签已配置文件的为，新发现的目录作为扩展存在
            var maxIndex = 0;
            if (currentLabels.Count > 0)
                maxIndex = currentLabels.Max(o => o.Index) + 1;

            foreach (var item in folderLabels)
            {
                var exitsItem = currentLabels.FirstOrDefault(o => o.Folder == item.Folder);
                if (exitsItem == null)
                {
                    item.Index = maxIndex;
                    maxIndex++;
                    currentLabels.Add(item);
                }
            }

            WriteLabelConfigFile(currentLabels, labelConfigFile);

            foreach(var labelInfo in currentLabels)
            {
                var labelFolder = Path.Combine(folder, labelInfo.Folder);
                // 只获得包含 jpg png bmp 等图片的文件
                var files = Directory.GetFiles(labelFolder).Where(s => s.IsImageFile()).ToArray();
                tf.enable_eager_execution();

                
                
            }

            throw new NotImplementedException();
        }

        /// <summary>
        /// 从文件中读取标签的配置信息
        /// 如果文件不存在，则返回空数组
        /// 如果解析失败，则会有异常
        /// </summary>
        /// <remarks>
        /// 
        /// csv 的文件格式为：
        /// 目录名,标签名称,标签索引
        /// 
        /// >> 标签名称可以为空，默认等同目录名
        /// >> 标签索引可以为空，默认按照目录名的顺序，从0开始
        /// 
        /// </remarks>
        /// <param name="configFileName"></param>
        /// <returns></returns>
        static List<LabelNameInfo> GetLabelFromConfigFile(string configFileName)
        {
            var labelNameInfos = new List<LabelNameInfo>();

            var fileInfo = new FileInfo(configFileName);
            if(!fileInfo.Exists)
            {
                return labelNameInfos;
            }
            
            if(fileInfo.Extension == "json"){
                // 按照json格式解析，未来会需要支持
                throw new Exception("Not implement json format");
            }

            // 按照csv格式进行解析
            var lines = File.ReadAllLines(configFileName);
            int index = 0;
            foreach(var item in lines){
                if(string.IsNullOrEmpty(item))
                    continue;

                var arr = item.Split(new []{','}, StringSplitOptions.RemoveEmptyEntries);
                LabelNameInfo labelInfo = new LabelNameInfo();
                labelInfo.Folder = arr[0];
                if(arr.Length > 1)
                    labelInfo.Label = arr[1];
                else
                    labelInfo.Label = labelInfo.Folder;
                
                if(arr.Length > 2)
                {
                    if(int.TryParse(arr[2], out var inputIndex))
                    {
                        labelInfo.Index = inputIndex;
                        index = Math.Max(inputIndex, index) + 1;
                    }
                    else{
                        throw new InvalidDataException($"Invalid index {arr[2]} label:{arr[0]} ");                        
                    }
                }else{
                    labelInfo.Index = index;
                }

                labelNameInfos.Add(labelInfo);
            }

            return labelNameInfos;
        }

        /// <summary>
        /// 将标签数据写入csv文件
        /// </summary>
        /// <param name="labels"></param>
        /// <param name="configFileName"></param>
        static void WriteLabelConfigFile(List<LabelNameInfo> labels, string configFileName){
            var lines = new List<string>();
            foreach(var item in labels){
                lines.Add($"{item.Folder},{item.Label},{item.Index}");
            }

            File.WriteAllLines(configFileName, lines);
        }

        static List<LabelNameInfo> GetLabelFromFolderName(string folder)
        {
            var baseFolderInfo = new DirectoryInfo(folder);
            if(!baseFolderInfo.Exists)
                throw new DirectoryNotFoundException($"Folder {folder} not found");

            var subFolders = baseFolderInfo.GetDirectories();
            if(subFolders.Length == 0)
                throw new DirectoryNotFoundException($"Folder {folder} not sub folder found");

            // get folderName and sort by name asc.
            // 按照目录名字做升序排列，并作为标签名称
            var subFolderNames = subFolders.Select(x => x.Name).OrderBy(o=>o).ToArray();

            var labelNameInfos = new List<LabelNameInfo>();
            for(var i = 0; i < subFolderNames.Length; i++)
            {
                var folderName = subFolderNames[i];
                var labelNameInfo = new LabelNameInfo();
                labelNameInfo.Folder = folderName;
                labelNameInfo.Label = folderName;
                labelNameInfos.Add(labelNameInfo);
            }

            return labelNameInfos;
        }

        class LabelNameInfo
        {
            /// <summary>
            /// 目录名称
            /// 对应到当前目录下有图片的子目录
            /// </summary>
            /// <value></value>
            public string Folder {get;set;}
            
            /// <summary>
            /// 标签名称
            /// 默认和目录名称一致，作为分类用的标签名称
            /// 但是也可以自定义（目录不方便用英文，标签可以用中文）
            /// </summary>
            /// <value></value>
            public string Label {get;set;}


            /// <summary>
            /// 标签索引
            /// 索引从0开始，在整个训练期间需要唯一
            /// 默认初始创建的时候按照目录名称自动创建
            /// 但是一旦开始训练，不建议再对目录对应的标签id做修改
            /// </summary>
            /// <value></value>
            public int Index {get;set;} = -1;
        }
    }
}