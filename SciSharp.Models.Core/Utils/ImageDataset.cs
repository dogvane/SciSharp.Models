using HDF.PInvoke;
using SciSharp.Models.Extend;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;
using Tensorflow;
using Tensorflow.Keras.Preprocessings;
using static Tensorflow.Binding;

namespace SciSharp.Models.Utils
{
    public static class ImageDataset
    {
        static readonly string[] IMAGE_ALLOWLIST_FORMATS = new string[] { ".jpg", ".png", ".bmp", ".gif", ".jpeg"};

        /// <summary>
        /// Generates a <c>tf.data.Dataset</c> from image files in a directory.
        /// 
        /// If your directory structure is:
        /// 
        /// main_directory/
        /// ...class_a/
        /// ......a_image_1.jpg
        /// ......a_image_2.jpg
        /// ...class_b/
        /// ......b_image_1.jpg
        /// ......b_image_2.jpg
        /// 
        /// Then calling <c>ImageDatasetFromDirectory(main_directory, labels: "inferred")</c> will return a <c>tf.data.Dataset</c> that yields batches of
        /// images from the subdirectories <c>class_a</c> and <c>class_b</c>, together with labels
        /// 0 and 1 (0 corresponding to <c>class_a</c> and 1 corresponding to <c>class_b</c>).
        /// 
        /// Supported image formats: jpeg, png, bmp, gif.
        /// Animated gifs are truncated to the first frame.
        /// </summary>
        /// <param name="directory">Directory where the data is located.
        ///     If <paramref name="labels"/> is "inferred", it should contain
        ///     subdirectories, each containing images for a class.
        ///     Otherwise, the directory structure is ignored.</param>
        /// <param name="labels">Either "inferred" (labels are generated from the directory structure),
        ///     None (no labels), or a list/tuple of integer labels of the same size as the number of
        ///     image files found in the directory. Labels should be sorted according
        ///     to the alphanumeric order of the image file paths
        ///     (obtained via <c>os.walk(directory)</c> in Python).</param>
        /// <param name="label_mode">String describing the encoding of <paramref name="labels"/>. Options are:
        ///     - "int": means that the labels are encoded as integers
        ///         (e.g. for <c>sparse_categorical_crossentropy</c> loss).
        ///     - "categorical" means that the labels are
        ///         encoded as a categorical vector
        ///         (e.g. for <c>categorical_crossentropy</c> loss).
        ///     - "binary" means that the labels (there can be only 2)
        ///         are encoded as <c>float32</c> scalars with values 0 or 1
        ///         (e.g. for <c>binary_crossentropy</c>).
        ///     - None (no labels).</param>
        /// <param name="classNames">Only valid if <paramref name="labels"/> is "inferred". This is the explicit
        ///     list of class names (must match names of subdirectories). Used
        ///     to control the order of the classes
        ///     (otherwise alphanumerical order is used).</param>
        /// <param name="colorMode">One of "grayscale", "rgb", "rgba". Default: "rgb".
        ///     Whether the images will be converted to
        ///     have 1, 3, or 4 channels.</param>
        /// <param name="batchSize">Size of the batches of data. Default: 32.
        ///     If <c>null</c>, the data will not be batched
        ///     (the dataset will yield individual samples).</param>
        /// <param name="imageSize">Size to resize images to after they are read from disk,
        ///     specified as <c>(height, width)</c>. Defaults to <c>(256, 256)</c>.
        ///     Since the pipeline processes batches of images that must all have
        ///     the same size, this must be provided.</param>
        /// <param name="shuffle">Whether to shuffle the data. Default: <c>true</c>.
        ///     If set to <c>false</c>, sorts the data in alphanumeric order.</param>
        /// <param name="seed">Optional random seed for shuffling and transformations.</param>
        /// <param name="validationSplit">Optional float between 0 and 1,
        ///     fraction of data to reserve for validation.</param>
        /// <param name="subset">Subset of the data to return.
        ///     One of "training", "validation" or "both".
        ///     Only used if <paramref name="validationSplit"/> is set.
        ///     When <paramref name="subset"/> is "both", the utility returns a tuple of two datasets
        ///     (the training and validation datasets respectively).</param>
        /// <param name="interpolation">String, the interpolation method used when resizing images.
        ///     Defaults to <c>bilinear</c>. Supports <c>bilinear</c>, <c>nearest</c>, <c>bicubic</c>,
        ///     <c>area</c>, <c>lanczos3</c>, <c>lanczos5</c>, <c>gaussian</c>, <c>mitchellcubic</c>.</param>
        /// <param name="followLinks">Whether to visit subdirectories pointed to by symlinks.
        ///     Defaults to <c>false</c>.</param>
        /// <param name="cropToAspectRatio">If <c>true</c>, resize the images without aspect
        ///     ratio distortion. When the original aspect ratio differs from the target
        ///     aspect ratio, the output image will be cropped so as to return the
        ///     largest possible window in the image (of size <paramref name="imageSize"/>) that matches
        ///     the target aspect ratio. By default (<c>cropToAspectRatio=false</c>),
        ///     aspect ratio may not be preserved.</param>
        /// <param name="kwargs">Legacy keyword arguments.</param>
        /// <returns>
        /// Returns a <see cref="tf.data.Dataset"/> object.
        /// 
        /// - If <paramref name="label_mode"/> is <c>null</c>, it yields <c>float32</c> tensors of shape
        ///   <c>(batch_size, image_size[0], image_size[1], num_channels)</c>, encoding images
        ///   (see below for rules regarding <c>num_channels</c>).
        /// - Otherwise, it yields a tuple <c>(images, labels)</c>, where <c>images</c>
        ///   has shape <c>(batch_size, image_size[0], image_size[1], num_channels)</c>,
        ///   and <c>labels</c> follows the format described below.
        ///
        /// Rules regarding labels format:
        ///
        /// - if <paramref name="label_mode"/> is <c>int</c>, the labels are an <c>int32</c> tensor of shape
        ///   <c>(batch_size,)</c>.
        /// - if <paramref name="label_mode"/> is <c>binary</c>, the labels are a <c>float32</c> tensor of
        ///   1s and 0s of shape <c>(batch_size, 1)</c>.
        /// - if <paramref name="label_mode"/> is <c>categorical</c>, the labels are a <c>float32</c> tensor
        ///   of shape <c>(batch_size, num_classes)</c>, representing a one-hot encoding of the class index.
        ///
        /// Rules regarding number of channels in the yielded images:
        ///
        /// - if <paramref name="colorMode"/> is <c>grayscale</c>,
        ///   there's 1 channel in the image tensors.
        /// - if <paramref name="colorMode"/> is <c>rgb</c>,
        ///   there are 3 channels in the image tensors.
        /// - if <paramref name="colorMode"/> is <c>rgba</c>,
        ///   there are 4 channels in the image tensors.
        ///   </returns>
        public static (IDatasetV2 train_dataset, IDatasetV2 val_dataset) image_dataset_from_directory(
            string directory,
            string labels = "inferred",
            string label_mode = "int",
            string[] classNames = null,
            string colorMode = "rgb",
            int? batchSize = null,
            (int Width, int Height) imageSize = default,
            bool shuffle = true,
            int? seed = 0,
            float? validationSplit = null,  // 验证集的比例（0~1）
            string subset = null,
            string interpolation = "bilinear",
            bool followLinks = false,
            bool crop_to_aspect_ratio = false,
            Dictionary<string, object> kwargs = null)
        {

            var colorModeParams = new[] { "rgb", "rgba", "grayscale" };
            if (!colorModeParams.Contains(colorMode))
            {
                throw new ArgumentException($"'colorMode' must be one of {{'rgb', 'rgba', 'grayscale'}}. Received: colorMode={colorMode}");
            }

            var labelModeParams = new[] { "int", "categorical", "binary", null };
            if (!labelModeParams.Contains(label_mode))
            {
                // "int":         当标签是整数值时使用。标签值直接作为整数处理，不进行任何转换或编码。适用于分类问题，其中标签是离散的整数类别。
                // "categorical": 当标签是多类别问题时使用。标签将被转换为独热编码（One - Hot Encoding），
                //                每个类别用一个二进制向量表示，向量中只有一个元素为1，其余元素为0。适用于多类别分类问题。
                //  "binary":     当标签是二分类问题时使用。标签将被转换为二进制值，通常是0和1，表示两个类别之一。适用于二分类问题。

                throw new ArgumentException($"'labelMode' argument must be one of 'int', 'categorical', 'binary', or null. Received: labelMode={label_mode}");
            }

            var subsetParams = new[] { "both", "training", "validation" };
            if (!subsetParams.Contains(subset))
            {
                throw new ArgumentException($"'subset' argument must be one of 'training', 'validation', or 'both'. Received: subset={subset}");
            }

            if (validationSplit  != null && (validationSplit < 0 || validationSplit > 1))
            {
                throw new ArgumentException($"'validationSplit' argument must be a float between 0 and 1. Received: validationSplit={validationSplit}");
            }

            Shape img_size;

            if (imageSize == default)
            {
                img_size = (256, 256);
            }
            else
            {
                img_size = (imageSize.Height, imageSize.Width);
            }

            //if (labels != "inferred" && labels != null && !new[] { "int", "categorical", "binary" }.Contains(labelMode))
            //{
            //    throw new ArgumentException($"'labels' argument should be a list/tuple of integer labels, of the same size as the number of image files in the target directory. If you wish to infer the labels from the subdirectory names in the target directory, pass 'labels=\"inferred\"'. If you wish to get a dataset that only contains images (no labels), pass 'labels=None'. Received: labels={labels}");
            //}

            if (labels != "inferred" && classNames != null)
            {
                throw new ArgumentException($"You can only pass 'classNames' if 'labels=\"inferred\"'. Received: labels={labels}, and class_names={classNames}");
            }

            //if (labels != null && labelMode == null)
            //{
            //    labels = null;
            //}

            int numChannels;
            switch (colorMode)
            {
                case "rgb":
                    numChannels = 3;
                    break;
                case "rgba":
                    numChannels = 4;
                    break;
                case "grayscale":
                    numChannels = 1;
                    break;
                default:
                    throw new ArgumentException($"'colorMode' must be one of {{'rgb', 'rgba', 'grayscale'}}. Received: colorMode={colorMode}");
            }

            interpolation = GetInterpolation(interpolation);

            CheckValidationSplitArg(validationSplit, subset, shuffle, seed);

            if (seed == null)
            {
                // seed = np.random.randint(1e6) 找时间比对一下这两个随机数谁块                
                seed = new Random().Next(1000000);
            }


            (var image_paths, var image_labels, var class_names) = DataSetUtil.index_directory(
                        directory,
                        labels: labels,
                        formats: IMAGE_ALLOWLIST_FORMATS,
                        classNames: classNames,
                        shuffle: shuffle,
                        seed: seed??0,
                        followLinks: followLinks);

            if (label_mode == "binary" && class_names.Length != 2)
            {
                throw new ArgumentException(
                    $"When passing `label_mode=\"binary\"`, there must be exactly 2 class_names. Received: class_names={class_names}");
            }

            if (subset == "both")
            {
                (var image_paths_train, var labels_train) = DataSetUtil.get_training_or_validation_split(image_paths, image_labels, validationSplit, "training");
                (var image_paths_val, var labels_val) = DataSetUtil.get_training_or_validation_split(image_paths, image_labels, validationSplit, "validation");
                var num_channels = 3;

                var train_dataset = paths_and_labels_to_dataset(image_paths_train, (imageSize.Height, imageSize.Width), num_channels: num_channels, labels: labels_train, label_mode: label_mode, num_classes: class_names.Length, interpolation: interpolation, crop_to_aspect_ratio: crop_to_aspect_ratio);
                var val_dataset = paths_and_labels_to_dataset(image_paths_val, (imageSize.Height, imageSize.Width), num_channels: num_channels, labels_val, label_mode, class_names.Length, interpolation, crop_to_aspect_ratio);

                //train_dataset.cache();
                //val_dataset.cache();

                train_dataset = train_dataset.batch(batchSize.Value);
                val_dataset = val_dataset.batch(batchSize.Value);
                train_dataset = train_dataset.prefetch(batchSize.Value);
                val_dataset = val_dataset.prefetch(batchSize.Value);

                if (batchSize != null)
                {
                    if (shuffle)
                    {
                        train_dataset = train_dataset.shuffle(batchSize.Value * 8, seed);
                    }


                }
                else
                {
                    if (shuffle)
                    {
                        train_dataset = train_dataset.shuffle(1024, seed);
                    }
                }

                train_dataset.class_names = class_names;
                val_dataset.class_names = class_names;

                return (train_dataset, val_dataset);
            }
            return (null, null);
        }


        public static string GetInterpolation(string interpolation)
        {
            interpolation = interpolation.ToLower();
            var set = ImageExtend.ConstantStringToHashSet<ResizeMethod>(true);
            if (!set.Contains(interpolation))
            {
                throw new NotImplementedException(
                    $"Value not recognized for `interpolation`: {interpolation}. Supported values " +
                    $"are: {string.Join(",", set)}"
);
            }

            return interpolation;
        }



        /// <summary>
        /// Raise errors in case of invalid argument values.
        /// </summary>
        /// <param name="validationSplit">Float between 0 and 1, fraction of data to reserve for validation.</param>
        /// <param name="subset">One of "training", "validation" or "both". Only used if validationSplit is set.</param>
        /// <param name="shuffle">Whether to shuffle the data. Either True or False.</param>
        /// <param name="seed">Random seed for shuffling and transformations.</param>
        /// <returns></returns>
        static public void CheckValidationSplitArg(float? validationSplit, string subset, bool shuffle, int? seed)
        {
            if (validationSplit == null || (validationSplit < 0 || validationSplit > 1))
            {
                throw new ArgumentException($"'validationSplit' must be between 0 and 1, received: {validationSplit}");
            }

            if ((validationSplit != null && string.IsNullOrEmpty(subset)) || (validationSplit == null && !string.IsNullOrEmpty(subset)))
            {
                throw new ArgumentException("If 'subset' is set, 'validationSplit' must be set, and inversely.");
            }

            if (subset != "training" && subset != "validation" && subset != "both" && subset != null)
            {
                throw new ArgumentException($"'subset' must be either 'training', 'validation' or 'both', received: {subset}");
            }

            //if (validationSplit != 0 && shuffle && (seed == 0 || seed == null))
            //{
            //    throw new ArgumentException("If using 'validationSplit' and shuffling the data, you must provide a 'seed' argument, to make sure that there is no overlap between the training and validation subset.");
            //}
        }


        public static IDatasetV2 paths_and_labels_to_dataset(Tensor image_paths, Shape image_size, int num_channels,
            string[] labels, string label_mode, int num_classes, string interpolation,
            bool crop_to_aspect_ratio = false)
        {
            var path_ds = tf.data.Dataset.from_tensor_slices(image_paths);
            var img_ds = path_ds.map(x => load_image(x, image_size, num_channels, interpolation, crop_to_aspect_ratio), num_parallel_calls: tf.data.AUTOTUNE);
            if (label_mode != null)
            {
                var label_ds = DataSetUtil.labels_to_dataset(labels, label_mode, num_classes);
                img_ds = tf.data.Dataset.zip(img_ds, label_ds);
            }
            return img_ds;
        }

        public static IDatasetV2 paths_and_labels_to_dataset(string[] image_paths, Shape image_size, int num_channels,
                                                                string[] labels, string label_mode, int num_classes,
                                                                string interpolation,bool crop_to_aspect_ratio = false)
            {
                var path_ds = tf.data.Dataset.from_tensor_slices(image_paths);
                var img_ds = path_ds.map(x => load_image(x, image_size, num_channels, interpolation, crop_to_aspect_ratio), num_parallel_calls: tf.data.AUTOTUNE);
                if (label_mode != null)
                {
                    var label_ds = DataSetUtil.labels_to_dataset(labels, label_mode, num_classes);
                    img_ds = tf.data.Dataset.zip(img_ds, label_ds);
                }
                return img_ds;
            }


            /// <summary>
            /// 
            /// </summary>
            /// <param name="path"></param>
            /// <param name="image_size">目标大小的整数元组</param>
            /// <param name="num_channels">图片通道数</param>
            /// <param name="interpolation">
            /// 用于调整大小的插值方法。支持 `bilinear`、`nearest`、`bicubic`、`area`、`lanczos3`、`lanczos5`、`gaussian`、`mitchellcubic`。默认为 `'bilinear'`。
            /// </param>
            /// <param name="crop_to_aspect_ratio">
            /// 将图像调整到目标大小，不扭曲纵横比。
            /// </param>
            /// <returns></returns>
            public static Tensor load_image(string path, Shape image_size, int num_channels, string interpolation, bool crop_to_aspect_ratio = false)
        {
            Tensor img = tf.io.read_file(path);
            return load_image_i(img, image_size, num_channels, interpolation, crop_to_aspect_ratio);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="path">已经将路径用 tf 处理过了的路径名</param>
        /// <param name="image_size">目标大小的整数元组</param>
        /// <param name="num_channels">图片通道数</param>
        /// <param name="interpolation">
        /// 用于调整大小的插值方法。支持 `bilinear`、`nearest`、`bicubic`、`area`、`lanczos3`、`lanczos5`、`gaussian`、`mitchellcubic`。默认为 `'bilinear'`。
        /// </param>
        /// <param name="crop_to_aspect_ratio">
        /// 将图像调整到目标大小，不扭曲纵横比。
        /// </param>
        /// <returns></returns>
        public static Tensor load_image(Tensor path, Shape image_size, int num_channels, string interpolation, bool crop_to_aspect_ratio = false)
        {
            Tensor img = tf.io.read_file(path);
            return load_image_i(img, image_size, num_channels, interpolation, crop_to_aspect_ratio);
        }

        private static Tensor load_image_i(Tensor img, Shape image_size, int num_channels, string interpolation, bool crop_to_aspect_ratio)
        {
            img = tf.image.decode_image(img, channels: num_channels, expand_animations: false);
            if (crop_to_aspect_ratio)
            {
                img = ImageUtil.smart_resize(img, image_size, num_channels, interpolation);
            }
            else
            {
                img = tf.image.resize(img, image_size, method: interpolation);
            }

            img.shape = (image_size[0], image_size[1], num_channels);
            return img;
        }

    }
}
