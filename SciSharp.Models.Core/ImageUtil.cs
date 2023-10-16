using SciSharp.Models.Extend;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using Tensorflow;
using Tensorflow.Keras.Preprocessings;
using Tensorflow.NumPy;
using static Tensorflow.Binding;

namespace SciSharp.Models
{
    public class ImageUtil
    {
        /// <summary>
        /// Read image from file
        /// channels is 3 for RGB image
        /// </summary>
        /// <param name="file_name"></param>
        /// <param name="input_height"></param>
        /// <param name="input_width"></param>
        /// <param name="channels"></param>
        /// <param name="input_mean"></param>
        /// <param name="input_std"></param>
        /// <returns></returns>
        public static Tensor ReadImageFromFile(string file_name,
            int input_height = 299,
            int input_width = 299,
            int channels = 3,
            int input_mean = 0,
            int input_std = 255)
        {
            tf.enable_eager_execution();
            var file_reader = tf.io.read_file(file_name, "file_reader");
            var image_reader = tf.image.decode_jpeg(file_reader, channels: channels, name: "jpeg_reader");
            var caster = tf.cast(image_reader, tf.float32);
            var dims_expander = tf.expand_dims(caster, 0);
            var resize = tf.constant(new int[] { input_height, input_width });            
            var ret_img = tf.image.resize_bilinear(dims_expander, resize);

            // 对图片做归一化处理
            // input_mean 可以调整图片的亮度
            // input_std 主要是归一化时，将图片限定到 0-1 之间，但如果使用input_mean调整了亮度，需考虑 input_std 的具体数值
            if(input_mean > 0)
                ret_img = tf.subtract(ret_img, new float[] { input_mean });

            if( input_std > 0)
                ret_img = tf.divide(ret_img, new float[] { input_std });

            tf.Context.restore_mode();
            return ret_img;
        }



        /// <summary>
        /// 从目录中的图像文件生成一个<see cref="tf.data.Dataset"/>。
        /// 
        /// 如果你的目录结构如下：
        /// 
        /// main_directory/
        /// ...class_a/
        /// ......a_image_1.jpg
        /// ......a_image_2.jpg
        /// ...class_b/
        /// ......b_image_1.jpg
        /// ......b_image_2.jpg
        /// 
        /// 调用 <see cref="ImageDatasetFromDirectory"/> 并传入 <paramref name="directory"/> 和 labels: "inferred"，将返回一个<see cref="tf.data.Dataset"/>，它将从子目录 <c>class_a</c> 和 <c>class_b</c> 中批量生成图像，同时返回标签 0 和 1（0 对应 <c>class_a</c>，1 对应 <c>class_b</c>）。
        /// 
        /// 支持的图像格式：jpeg、png、bmp、gif。
        /// 动态gif将被截断为第一帧。
        /// </summary>
        /// <param name="directory">数据所在的目录。
        ///     如果 <paramref name="labels"/> 为 "inferred"，它应包含子目录，每个子目录包含一个类别的图像。
        ///     否则，目录结构将被忽略。</param>
        /// <param name="labels">标签类型。可以是 "inferred"（从目录结构中生成标签）、<c>null</c>（没有标签），
        ///     或者与图像文件数量相同的整数标签列表/元组。
        ///     标签应根据图像文件路径的字母数字顺序进行排序
        ///     （使用 <c>os.walk(directory)</c> 在Python中获取）。</param>
        /// <param name="labelMode">描述 <paramref name="labels"/> 编码方式的字符串。可选项包括：
        ///     - "int"：标签以整数编码（例如用于 <c>sparse_categorical_crossentropy</c> 损失）。
        ///     - "categorical"：标签以分类向量编码（例如用于 <c>categorical_crossentropy</c> 损失）。
        ///     - "binary"：标签（只能有2个）以取值为0或1的 <c>float32</c> 标量编码
        ///         （例如用于 <c>binary_crossentropy</c> 损失）。
        ///     - <c>null</c>：没有标签。</param>
        /// <param name="classNames">仅在 <paramref name="labels"/> 为 "inferred" 时有效。这是类名的显式列表（必须与子目录的名称相匹配）。用于控制类别的顺序（否则将使用字母数字顺序）。</param>
        /// <param name="colorMode">颜色模式。可选值为 "grayscale"、"rgb"、"rgba"。默认为 "rgb"。表示图像将被转换为具有1、3或4个通道。</param>
        /// <param name="batchSize">数据的批量大小。默认为32。如果为<c>null</c>，则不进行批处理（数据将逐个样本生成）。</param>
        /// <param name="imageSize">从磁盘读取图像后调整大小的尺寸，格式为（高度，宽度）。默认为（256, 256）。
        ///     由于管道处理的是具有相同大小的图像批次，因此必须提供此参数。</param>
        /// <param name="shuffle">是否对数据进行洗牌。默认为 <c>true</c>。如果设置为 <c>false</c>，则按字母数字顺序对数据进行排序。</param>
        /// <param name="seed">用于洗牌和变换的可选随机种子。</param>
        /// <param name="validationSplit">可选的介于0和1之间的浮点数，用于保留验证数据的比例。</param>
        /// <param name="subset">要返回的数据子集。可选值为 "training"、"validation" 或 "both"。
        ///     仅在设置了 <paramref name="validationSplit"/> 时使用。
        ///     当 <paramref name="subset"/> 为 "both" 时，返回两个数据集的元组（分别是训练数据集和验证数据集）。</param>
        /// <param name="interpolation">调整图像大小时使用的插值方法。默认为 "bilinear"。支持 <c>bilinear</c>、<c>nearest</c>、<c>bicubic</c>、
        ///     <c>area</c>、<c>lanczos3</c>、<c>lanczos5</c>、<c>gaussian</c>、<c>mitchellcubic</c>。</param>
        /// <param name="followLinks">是否访问符号链接指向的子目录。默认为 <c>false</c>。</param>
        /// <param name="cropToAspectRatio">如果为 <c>true</c>，则调整图像大小而不失真纵横比。
        ///     当原始纵横比与目标纵横比不同时，将裁剪输出图像以返回与目标纵横比匹配的最大可能窗口（大小为 <paramref name="imageSize"/>）。
        ///     默认情况下（<c>cropToAspectRatio=false</c>），可能不保留纵横比。</param>
        /// <param name="kwargs">旧版的关键字参数。</param>
        /// <returns>
        /// 返回一个<see cref="tf.data.Dataset"/>对象。
        /// 
        /// - 如果 <paramref name="labelMode"/> 为 <c>null</c>，它将生成形状为 `(batch_size, image_size[0], image_size[1], num_channels)` 的 `float32` 张量，
        ///   表示图像（关于 `num_channels` 的规则见下文）。
        /// - 否则，它将生成一个元组 `(images, labels)`，其中 `images` 的形状为 `(batch_size, image_size[0], image_size[1], num_channels)`，
        ///   并且 `labels` 遵循下述格式。

        /// 关于标签格式的规则：
        /// 
        /// - 如果 <paramref name="labelMode"/> 为 `int`，标签是形状为 `(batch_size,)` 的 `int32` 张量。
        /// - 如果 <paramref name="labelMode"/> 为 `binary`，标签是形状为 `(batch_size, 1)` 的取值为 1 和 0 的 `float32` 张量。
        /// - 如果 <paramref name="labelMode"/> 为 `categorical`，标签是形状为 `(batch_size, num_classes)` 的 `float32` 张量，
        ///   表示类别索引的独热编码。

        /// 关于生成的图像中通道数的规则：
        /// 
        /// - 如果 <paramref name="colorMode"/> 为 `grayscale`，图像张量中有 1 个通道。
        /// - 如果 <paramref name="colorMode"/> 为 `rgb`，图像张量中有 3 个通道。
        /// - 如果 <paramref name="colorMode"/> 为 `rgba`，图像张量中有 4 个通道。
        /// </returns>
        public static IEnumerable<(string ImagePath, int Label)> ImageDatasetFromDirectory2(
            string directory,
            string labels = "inferred",
            string labelMode = "int",
            List<string> classNames = null,
            string colorMode = "rgb",
            int batchSize = 32,
            (int Width, int Height) imageSize = default,
            bool shuffle = true,
            int? seed = null,
            double? validationSplit = null,
            string subset = null,
            string interpolation = "bilinear",
            bool followLinks = false,
            bool cropToAspectRatio = false,
            Dictionary<string, object> kwargs = null)
        {
            throw new NotSupportedException();
        }
    }
}
