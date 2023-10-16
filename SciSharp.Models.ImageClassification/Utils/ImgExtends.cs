using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace SciSharp.Models.ImageClassification
{
    public static class ImgExtends
    {
        /// <summary>
        /// check file is image
        /// </summary>
        /// <param name="fileName"></param>
        /// <returns></returns>
        public static bool IsImageFile(this string fileName)
        {
            return fileName.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase)
                || fileName.EndsWith(".png", StringComparison.OrdinalIgnoreCase)
                || fileName.EndsWith(".bmp", StringComparison.OrdinalIgnoreCase);
        }
    }
}