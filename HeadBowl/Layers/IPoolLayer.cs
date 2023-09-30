namespace HeadBowl.Layers
{
    public interface IPoolLayer<T> : ILayer<T>
    {
        int Stride { get; }
        int FilterSize { get; }
        PoolingType Type { get; }
    }


    //internal abstract class PoolLayer_64bit : IPoolLayer<double>
    //{

    //}
}
