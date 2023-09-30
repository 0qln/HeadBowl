namespace HeadBowl.Layers
{
    public enum PoolingType 
    {
        /// <summary>
        /// The Max Pooling layer summarizes the features in a region represented by the maximum value in that region. 
        /// Max Pooling is more suitable for images with a dark background as it will select brighter pixels in a region 
        /// of the input image. Hence, with Max Pooling we retain the most significant features of the feature map, 
        /// and the resulting image becomes sharper than the input image.
        /// </summary>
        MaxPooling,

        /// <summary>
        /// The Min Pooling layer summarizes the features in a region represented by the minimum value in that region. 
        /// Contrary to Max Pooling in CNN, this type is mainly used for images with a light background to focus on 
        /// darker pixels.
        /// </summary>
        MinPooling,

        /// <summary>
        /// The average pooling summarizes the features in a region represented by the average value of that region. 
        /// With average pooling, the harsh edges of a picture are smoothened, and this type of pooling layer can used when 
        /// harsh edges can be ignored.
        /// </summary>
        AvgPooling,

        /// <summary>
        /// The pooling technique reduces each feature map channel to a single value. This value depends on the type 
        /// of global pooling, which can be any of the previously explained pooling types. In other words, applying 
        /// global pooling is similar to using a filter of the exact dimensions of the feature map.
        /// </summary>
        GlobalPooling,
    }

}
