using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.Caching;
using System.Text;
using Tensorflow;

namespace SciSharp.Models.Extend
{
    public static class ImageExtend
    {
        private static readonly MemoryCache ConstantStringToHashSetCache = new MemoryCache("ConstantStringToHashSet");

        /// <summary>
        /// 将类型下的常量字符串，转为一个hash字符串
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="toLower">
        /// 是否转为小写，默认为false，不对常量做处理
        /// 如果为 true， 则将常量转为小写，方便后续的判断使用
        /// </param>
        /// <returns></returns>
        public static HashSet<string> ConstantStringToHashSet<T>(bool toLower = false)
        {
            var type = typeof(T);

            string cacheKey = type.FullName;
            HashSet<string> constantStrings = ConstantStringToHashSetCache.Get(cacheKey) as HashSet<string>;

            if (constantStrings == null)
            {
                constantStrings = new HashSet<string>();

                FieldInfo[] fields = type.GetFields(BindingFlags.Public | BindingFlags.Static | BindingFlags.FlattenHierarchy);

                foreach (FieldInfo field in fields)
                {
                    if (field.IsLiteral && !field.IsInitOnly && field.FieldType == typeof(string))
                    {
                        string constantString = (string)field.GetValue(null);
                        if(toLower)
                            constantStrings.Add(constantString.ToLower());
                        else
                            constantStrings.Add(constantString);
                    }
                }

                ConstantStringToHashSetCache.Set(cacheKey, constantStrings, DateTimeOffset.MaxValue);
            }

            return constantStrings;
        }

        /// <summary>
        /// 对列表洗牌
        /// 这里允许传入多个列表，必须保证列表的维度保持一致
        /// 在洗牌的时候，会同同步洗所有列表
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="rand"></param>
        /// <param name="shuffleArray">允许多个列表传入</param>
        public static void Shuffle<T>(this Random rand, params List<T>[] shuffleArray)
        {
            if (shuffleArray == null || shuffleArray.Length == 0)
                throw new ArgumentException("shuffleArray not empty.");

            var len = shuffleArray[0].Count;
            // 检查维度
            if(shuffleArray.Any(o=>o.Count != len))
            {
                throw new Exception("The dimensions of the lists in shuffleArray are inconsistent.");
            }

            for (int i = 0; i < len; i++)
            {
                var index = rand.Next(len);
                for (int j = 0; j < shuffleArray.Length; j++)
                {
                    var list = shuffleArray[j];
                    var temp = list[i];
                    list[i] = list[index];
                    list[index] = temp;
                }
            }
        }
    }
}
