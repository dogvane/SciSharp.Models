using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.NumPy;
using Tensorflow;
using Tensorflow.Keras.Preprocessings;
using static Tensorflow.Binding;


namespace SciSharp.Models.Utils
{
    internal class ImageUtil
    {
      

        /// <summary>
        /// 将图像调整到目标大小，不扭曲纵横比。
        /// 
        /// 警告：不推荐使用 `tf.keras.preprocessing.image.smart_resize`。建议使用 `tf.keras.layers.Resizing`，
        /// 它提供了与预处理层相同的功能，并添加了 `tf.RaggedTensor` 支持。
        /// 有关预处理层的概述，请参阅[预处理层指南](https://www.tensorflow.org/guide/keras/preprocessing_layers)。
        /// 
        /// TensorFlow 图像数据集通常包含具有不同大小的图像。然而，这些图像在传递给 Keras 层之前需要进行批处理。
        /// 要进行批处理，图像需要具有相同的高度和宽度。
        /// 
        /// 您可以简单地执行以下操作：
        /// 
        /// ```python
        /// size = (200, 200)
        /// ds = ds.map(lambda img: tf.image.resize(img, size))
        /// ```
        /// 
        /// 但是，如果这样做，会扭曲图像的纵横比，因为一般情况下它们的纵横比与 `size` 不同。
        /// 在许多情况下，这是可以接受的，但并非始终如此（例如，对于 GANs，这可能是一个问题）。
        /// 
        /// 请注意，将参数 `preserve_aspect_ratio=True` 传递给 `resize` 将保持纵横比，但代价是不再遵守提供的目标大小。
        /// 因为 `tf.image.resize` 不裁剪图像，所以输出图像仍然具有不同的大小。
        /// 
        /// 这就需要：
        /// 
        /// ```python
        /// size = (200, 200)
        /// ds = ds.map(lambda img: smart_resize(img, size))
        /// ```
        /// 
        /// 输出图像实际上将为 `(200, 200)`，并且不会被扭曲。相反，不适合目标大小的图像部分将被裁剪掉。
        /// 
        /// 调整大小的过程如下：
        /// 
        /// 1. 获取具有与目标大小相同纵横比的图像的最大中心裁剪。
        ///     例如，如果 `size=(200, 200)`，输入图像的大小为 `(340, 500)`，
        ///     我们会获取一个沿宽度居中的 `(340, 340)` 裁剪。
        ///     
        /// 2. 将裁剪后的图像调整为目标大小。
        ///     在上面的示例中，我们将 `(340, 340)` 的裁剪调整为 `(200, 200)`。
        /// 
        /// </summary>
        /// <param name="img">
        /// 输入图像或图像批处理（作为张量或NumPy数组）。
        /// 必须是格式为`(height, width, channels)`或`(batch_size, height, width, channels)`的形式。
        /// </param>
        /// <param name="size">目标大小的整数元组`(height, width)`。</param>
        /// <param name="interpolation">
        /// 用于调整大小的插值方法。支持`bilinear`、`nearest`、`bicubic`、`area`、`lanczos3`、`lanczos5`、`gaussian`、`mitchellcubic`。
        /// 默认为`'bilinear'`。
        /// </param>
        /// <returns>形状为`(size[0], size[1], channels)`的数组。如果输入图像是NumPy数组，则输出为NumPy数组；如果输入图像是TF张量，则输出为TF张量。</returns>
        public static Tensor smart_resize(Tensor img, Shape size, int num_channels, string interpolation = "bilinear")
        {
            if (size.size != 2)
                throw new ValueError($"Expected `size` to be a tuple of 2 integers, but got: {size}.");

            if (img.shape.rank < 3 || img.shape.rank > 4)
                throw new ValueError($"Expected an image array with shape `(height, width, channels)`, or `(batch_size, height, width, channels)`, but got input with incorrect rank, of shape {img.shape}.");

            Tensor shape = tf.shape(img);
            var height = shape[-3];
            var width = shape[-2];
            var target_height = size[0];
            var target_width = size[1];
            

            var crop_height = tf.cast(tf.cast(width * target_height, TF_DataType.TF_FLOAT) / target_width, TF_DataType.TF_INT32);
            var crop_width = tf.cast(tf.cast(height * target_width, TF_DataType.TF_FLOAT) / target_height, TF_DataType.TF_INT32);

            crop_height = tf.minimum(height, crop_height);
            crop_width = tf.minimum(width, crop_width);

            var crop_box_hstart = tf.cast(tf.cast(height - crop_height, TF_DataType.TF_FLOAT) / 2, TF_DataType.TF_INT32);
            var crop_box_wstart = tf.cast(tf.cast(width - crop_width, TF_DataType.TF_FLOAT) / 2, TF_DataType.TF_INT32);

            Tensor crop_box_start, crop_box_size;
            if (img.shape.rank == 4)
            {
                crop_box_start = tf.stack(new Tensor[] { tf.constant(0), crop_box_hstart, crop_box_wstart, tf.constant(0) });
                crop_box_size = tf.stack(new Tensor[] { tf.constant(-1), crop_height, crop_width, tf.constant(-1) });
            }
            else
            {
                crop_box_start = tf.stack(new Tensor[] { crop_box_hstart, crop_box_wstart, tf.constant(0) });
                crop_box_size = tf.stack(new Tensor[] { crop_height, crop_width, tf.constant(-1) });
            }

            // img = tf.slice(img, crop_box_start, crop_box_size)
            img = array_ops.slice(img, crop_box_start, crop_box_size);
            img = tf.image.resize(img, size, interpolation);

            if (img.shape.rank == 4)
                img.set_shape(new Shape(-1, -1, -1, num_channels));
            if (img.shape.rank == 3)
                img.set_shape(new Shape(-1, -1, num_channels));

            return img;
        }
    }
}
