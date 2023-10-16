using SciSharp.Models.Extend;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;
using Tensorflow;
using Tensorflow.Keras.Preprocessings;
using static Tensorflow.Binding;

namespace SciSharp.Models.Utils
{
    /// <summary>
    /// 数据集用的一些辅助类
    /// </summary>
    public static class DataSetUtil
    {
        public class IndexResult
        {
            public List<string> FilePaths { get; set; }
            public List<int> Labels { get; set; }
            public List<string> ClassNames { get; set; }
        }

        /// <summary>
        /// 在指定目录中列出所有文件，并附带它们的标签。
        /// </summary>
        /// <param name="directory">数据所在的目录。
        /// 如果"labels"为"inferred"，则该目录应包含子目录，每个子目录包含一个类别的文件。
        /// 否则，将忽略目录结构。</param>
        /// <param name="labels">标签类型，可以是以下之一：
        /// - "inferred"（自己'推断'，从目录结构中生成标签）
        /// - None（无标签）
        /// - 整数标签的列表/元组，与目录中有效文件的数量相同。
        /// 标签应按照图像文件路径的字母数字顺序进行排序
        /// （在Python中通过`os.walk(directory)`获得）。</param>
        /// <param name="formats">允许索引的文件扩展名白名单（例如".jpg"、".txt"）。</param>
        /// <param name="class_names">仅当"labels"为"inferred"时有效。
        /// 这是类别名称的显式列表（必须与子目录的名称匹配）。
        /// 用于控制类别的顺序（否则将使用字母数字顺序）。</param>
        /// <param name="shuffle">是否对数据进行洗牌。默认值为True。
        /// 如果设置为False，则按字母数字顺序排序数据。</param>
        /// <param name="seed">用于洗牌的可选随机种子。</param>
        /// <param name="follow_links">是否访问符号链接指向的子目录。</param>
        /// <returns>元组 (file_paths, labels, class_names)。
        /// - file_paths：文件路径列表（字符串）。
        /// - labels：匹配的整数标签列表（与file_paths长度相同）。
        /// - class_names：对应这些标签的类别名称列表，按顺序排列。</returns>
        public static (string[] image_paths, string[] labels, string[] class_names) index_directory(
            string directory,
            string labels = "inferred",
            string[] formats = null,
            string[] classNames = null,
            bool shuffle = true,
            int seed = 0,
            bool followLinks = false)
        {
            // 看了半天的python，最后，还是根据接口注释，直接重写，目前整理一下的逻辑细节有：
            // 1. labels 为 inferred 时，会根据目录结构生成标签，否则，标签用数字作为标签，并且数字的排序规则，按照目录名字母升序排序
            // 2. formats 为 null 时，会默认使用所有的图片格式(暂定格式有：jpg png)，否则，只会使用指定的图片格式（文档没指名允许单个后缀还是多后缀，代码里看上去是单一后缀）
            // 3. class_names 为 null 时，会默认使用目录名字作为类别名，否则，会使用指定的类别名，但是，如果指定的类别名数量和目录数量不一致，会抛出异常
            // 4. shuffle 为 true 时，会对数据进行洗牌，否则，按照字母数字顺序排序数据(这里的字母数字顺序，或者有指定的class_names顺序)
            // 5. seed 为 0 时，会使用默认的随机种子，否则，使用指定的随机种子
            // 6. follow_links 为 true 时，会访问符号链接指向的子目录，否则，不会访问符号链接指向的子目录（功能待议，这个参数的实际效果还不太明确）

            var dirInfo = new DirectoryInfo(directory);
            if (!dirInfo.Exists)
            {
                throw new Exception($"not find directory:{directory}");
            }

            List<DirectoryInfo> subdirs = new List<DirectoryInfo>(dirInfo.GetDirectories().OrderBy(o=>o.Name));
            var idMap = new Dictionary<string, int>();

            if(labels == "inferred")
            {
                var id = 0;
                foreach(var labelName in subdirs.Select(o=>o.Name).OrderBy(o=>o))
                {
                    idMap.Add(labelName, id);
                    id++;
                }
            }

            if (classNames != null)
            {
                // classNames 时，需要保证和当前的所有目录一致 
                HashSet<string> set_subdirs = new HashSet<string>(subdirs.Select(o => o.Name));
                var id = 0;
                foreach (var labelName in set_subdirs)
                {
                    idMap.Add(labelName, id);
                    id++;
                }

                if (set_subdirs.SetEquals(classNames))
                {
                    throw new ArgumentException(
                            $"The 'class_names' passed did not match the names of the subdirectories of the target directory. " +
                            $"Expected: {string.Join(", ", set_subdirs)}, but received: {string.Join(", ", classNames)}");
                }
            }
            else
            {
                if (idMap.Count == 0)
                    classNames = subdirs.Select(o => o.Name).OrderBy(o => o).ToArray();
                else
                    classNames = idMap.Values.OrderBy(o => o).Select(o => o.ToString()).ToArray();
            }

            if(formats == null)
            {
                formats = new string[] { ".jpg", ".png", ".bmp", ".gif", ".jpeg"};
            }
            
            var filenames = new List<string>();
            var file_labels = new List<string>();

            foreach (var dir in subdirs)
            {
                var label_name = dir.Name;
                if (idMap.Count > 0)
                    label_name = idMap[label_name].ToString();

                var relative_path = get_dir_relative_files(dir, formats);
                foreach(var f in relative_path)
                {
                    filenames.add(Path.Combine(directory, f));
                    file_labels.add(label_name);
                }
            }

            if (shuffle)
            {
                var rand = seed != 0 ? new Random(seed) : new Random();
                rand.Shuffle(filenames, file_labels);
            }

            return (filenames.ToArray(), file_labels.ToArray(), classNames.ToArray());
        }

        /// <summary>
        /// 获得目录文件
        /// 文件会转为相对上级目录的相对目录文件格式
        /// 例如： c:\classes\flower\f_1.jpg
        /// 返回： flower\f_1.jpg
        /// </summary>
        /// <param name="dirInfo"></param>
        /// <param name="formats"></param>
        /// <returns></returns>
        private static string[] get_dir_relative_files(DirectoryInfo dirInfo,  string[] formats)
        {
            var relative_paths = new List<string>();

            var parentFolder = dirInfo.Parent.FullName;
            foreach (var file in dirInfo.GetFiles().OrderBy(o => o.Name))
            {
                if (formats.Any(o => o.Equals(file.Extension, StringComparison.OrdinalIgnoreCase)))
                {
                    var rp = file.FullName.Substring(parentFolder.Length + 1);
                    relative_paths.add(rp);
                }
            }

            return relative_paths.ToArray();
        }

        /// <summary>
        /// 将列表/元组标签转换为 TensorFlow Dataset。
        /// 对应： dataset_utils.labels_to_dataset 方法
        /// </summary>
        /// <param name="labels">要转换为 TensorFlow Dataset 的标签列表/元组。</param>
        /// <param name="labelMode">描述“labels”编码方式的字符串。选项有：
        /// - "binary" 表示标签（只有2个标签）被编码为取值为0或1的“float32”标量（例如用于"binary_crossentropy"）。
        /// - "categorical" 表示标签被映射为分类向量（例如用于"categorical_crossentropy"损失）。</param>
        /// <param name="numClasses">标签的类别数量。</param>
        /// <returns>一个 TensorFlow Dataset 实例。</returns>
        public static IDatasetV2 labels_to_dataset(string[] labels, string labelMode, int numClasses)
        {
            var label_ds = tf.data.Dataset.from_tensor_slices(labels.Select(o=>int.Parse(o)).ToArray());

            if (labelMode == "binary")
            {
                label_ds = label_ds.map(x => tf.expand_dims(tf.cast(x, TF_DataType.TF_FLOAT), -1), num_parallel_calls: tf.data.AUTOTUNE);
            }
            else if (labelMode == "categorical")
            {
                label_ds = label_ds.map(x => tf.one_hot(x, numClasses), num_parallel_calls: tf.data.AUTOTUNE);
            }

            return label_ds;
        }

        /// <summary>
        /// 切分训练集和验证集
        /// </summary>
        /// <param name="samples">元素的列表。</param>
        /// <param name="labels">对应标签的列表。</param>
        /// <param name="validation_split">浮点数，保留用于验证的数据比例。</param>
        /// <param name="subset">要返回的数据子集。可以是 "training"、"validation" 或 null。如果是 null，则返回所有数据。</param>
        /// <returns>元组 (samples, labels)，可能限制为指定的子集。</returns>
        public static (T[] samples, T[] labels) get_training_or_validation_split<T>(T[] samples, T[] labels, float? validation_split, string subset = null)
        {
            if (validation_split == null)
                return (samples, labels);

            var num_val_samples = (int)(validation_split * samples.Length);
            if (subset == "training")
            {
                Console.WriteLine($"Using {samples.Length - num_val_samples} files for training.");
                samples = samples.Take(samples.Length - num_val_samples).ToArray();
                labels = labels.Take(labels.Length - num_val_samples).ToArray();
            }
            else if (subset == "validation")
            {
                Console.WriteLine($"Using {num_val_samples} files for validation.");
                samples = samples.Skip(samples.Length - num_val_samples).Take(num_val_samples).ToArray();
                labels = labels.Skip(labels.Length - num_val_samples).Take(num_val_samples).ToArray();
            }
            
            return (samples, labels);
        }
    }
}
