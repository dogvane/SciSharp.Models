using SciSharp.Models;
using SciSharp.Models.ImageClassification.Zoo;
using System;
using System.IO;
using System.Linq;
using Tensorflow.Keras;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow.NumPy;
using SciSharp.Models.ImageClassification;
using Tensorflow;
using System.Collections.Generic;

namespace Models.Run
{
    class Program
    {
        static void Main(string[] args)
        {
            tf.Context.Config.GpuOptions.AllowGrowth = true;

            //dumpOpCode();
            //return;

            // KereaTest();

            // return;

            // FixImgRunTest();

            // RunTrain();

            RunTrainFlower();

            Console.WriteLine("Completed.");
        }

        static public void FixImgRunTest()
        {
            var img_size = 112;

            //var imgFolder = "V:\\\\Data\\\\mmlab_data\\\\avi_person_out\\\\classes";
            var imgFolder = "D:\\data\\flower_photos";
            var image_size = (img_size, img_size);
            var batch_size = 128;

            var trainData = keras.preprocessing.image_dataset_from_directory(
                imgFolder,
                image_size: image_size,
                batch_size: batch_size,
                subset: "training",
                validation_split: 0.3f
                );

            var validationData = keras.preprocessing.image_dataset_from_directory(
                imgFolder,
                image_size: image_size,
                batch_size: batch_size,
                subset: "validation",
                validation_split: 0.3f
                );

            var classes_num = trainData.class_names.Length;

            // two dimensions input with unknown batchsize
            var input = tf.keras.layers.Input((img_size, img_size, 3)); // mock image data            
            var x = input;

            x = keras.layers.Conv2D(32, 3, 1, "same", activation: "relu").Apply(x);
            x = keras.layers.Conv2D(32, kernel_size: (1, 1), strides: 1, padding: "same", activation: "relu").Apply(x);
            x = tf.keras.layers.DepthwiseConv2D(3, 1, "same", use_bias: false).Apply(x);
            x = tf.keras.layers.Normalization().Apply(x);
            //x = tf.keras.activations.Relu.Apply(x);
            x = tf.keras.layers.Flatten().Apply(x);
            var output = tf.keras.layers.Dense(classes_num).Apply(x);

            tf.Context.Config.GpuOptions.AllowGrowth = true;

            var model = tf.keras.Model(input, output);
            model.compile(tf.keras.optimizers.SGD(), tf.keras.losses.SparseCategoricalCrossentropy());
            model.summary();

            model.fit(
                trainData,
                validation_data: validationData,
                validation_step: 10,
                epochs: 10,
                batch_size: 10,
                workers: 1,
                use_multiprocessing: false);
        }

        static void dumpOpCode()
        {
            var d = new List<string>();
            var buffer = new Tensorflow.Buffer(c_api.TF_GetAllOpList());
            var op_list = OpList.Parser.ParseFrom(buffer.ToArray());
            foreach (var op_def in op_list.Op)
            {
                d.Add(op_def.ToString());
            }

            File.WriteAllLines("opcode.txt", d);
        }
        static void KereaTest()
        {
            ILayer l = keras.layers.Normalization();
            ILayer d = keras.layers.Dense(10);

            var input = tf.ones((3, 3), name: "input");
            
            //var block1 = keras.Sequential(new[] { l, d });
            //print(block1.Apply(input));

            var block2 = keras.Sequential(new[] {
                new BlocksLayer(new[] { l }),
                new BlocksLayer(new[] { d })
            });

            var output = block2.Apply(input);

            print(output);
        }

        static void RunTrainFlower()
        {
            var config = new FolderClassificationConfig();
            if (Environment.OSVersion.Platform == PlatformID.Win32NT)
                config.BaseFolder = "D:\\data\\flower_photos";
            else
                config.BaseFolder = "/mnt/d/data/flower_photos";

            config.DataDir = "";
            var img_size = 224;

            config.InputShape = (img_size, img_size);

            config.BatchSize = 24;
            config.ValidationStep = 5;
            config.Epoch = 50;

            var model = new VGG();
            //var model = new AlexNet();

            var classifier = new FolderClassification(config, model);

            config.WeightsPath = $"{model.GetType().Name}_{img_size}x{img_size}_weights.ckpt";

            classifier.Train();
        }

        static void RunTrain()
        {
            var config = new FolderClassificationConfig();
            if (Environment.OSVersion.Platform == PlatformID.Win32NT)
                config.BaseFolder = "V:\\Data\\mmlab_data\\avi_person_out";
            else
                config.BaseFolder = "/mnt/v/data/mmlab_data/avi_person_out";
            
            config.DataDir = "classes";
            var img_size = 224;

            config.InputShape = (img_size, img_size);

            config.BatchSize = 32;
            config.ValidationStep = 1;
            config.Epoch = 50;

            var model = new VGG();
            //var model = new AlexNet();

            var classifier = new FolderClassification(config, model);

            config.WeightsPath = $"{model.GetType().Name}_{img_size}x{img_size}_weights.ckpt";

            classifier.Train();
        }

        static void RunPredictFolder()
        {
            var config = new FolderClassificationConfig();
            if (Environment.OSVersion.Platform == PlatformID.Win32NT)
                config.BaseFolder = "V:\\Data\\mmlab_data\\avi_person_out";
            else
                config.BaseFolder = "/mnt/v/data/mmlab_data/avi_person_out";

            var img_size = 224;

            config.InputShape = (img_size, img_size);

            var model = new MobilenetV2();
            //var model = new AlexNet();

            var classifier = new FolderClassification(config, model);

            config.WeightsPath = $"{model.GetType().Name}_{img_size}x{img_size}_weights.ckpt";

            var files = Directory.GetFiles(Path.Combine(config.BaseFolder, "person"));
            var i = 6700;
            var batchSize = 16;
            int pCount = 0;
            int mCount = 0;

            string outFolder = "V:\\Data\\mmlab_data\\avi_person_out\\deme_out";
            foreach (var p in classifier.ClassNames)
            {
                Directory.CreateDirectory(Path.Combine(outFolder, p));
            }

            while (i < files.Length)
            {
                var fs = files.Skip(i).Take(batchSize).ToArray();
                i += batchSize;

                var ret = classifier.Predict(fs);

                for (var j = 0; j < ret.Length; j++)
                {
                    if (ret[j].Probability > 70.0f && ret[j].Probability < 90.0f && ret[j].Label == "gufeng")
                    {
                        var outFile = Path.Combine(outFolder, ret[j].Label, new FileInfo(fs[j]).Name);

                        File.Copy(fs[j], outFile, true);
                        pCount++;
                        if (pCount > 100)
                        {
                            Console.WriteLine($"find:{100.0 * pCount / (pCount + mCount)}");
                            return;
                        }
                    }
                    else
                    {
                        mCount++;
                    }
                }
            }
        
        }
    }
}
