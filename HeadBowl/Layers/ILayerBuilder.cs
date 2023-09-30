namespace HeadBowl.Layers
{
    public interface ILayerBuilder<T>
    {
        public int Size { get; }
        internal void SetNext(ILayer<T>? layer);
        internal void SetPrev(ILayer<T>? layer);
        public ILayer<T> Build();
        public ILayer<T> Instance();
    }

    public abstract class LayerBuilderBase<TPrecision> : ILayerBuilder<TPrecision>
    {
        public int Size => _instance.Size;

#pragma warning disable CS8618
        protected ILayer<TPrecision> _instance;
#pragma warning restore CS8618

        protected ILayer<TPrecision>? _next, _prev;
        

        void ILayerBuilder<TPrecision>.SetNext(ILayer<TPrecision>? layer) => _next = layer;
        void ILayerBuilder<TPrecision>.SetPrev(ILayer<TPrecision>? layer) => _prev = layer;

        ILayer<TPrecision> ILayerBuilder<TPrecision>.Build()
        {
            _instance._InitInNet(_prev, _next);
            return _instance;
        }
        ILayer<TPrecision> ILayerBuilder<TPrecision>.Instance()
        {
            return _instance;
        }
    }

    public class FullyConnectedLayer<TPrecision> : LayerBuilderBase<TPrecision>
    {
        public FullyConnectedLayer(ActivationType activation, int size)
        {
            _instance = activation switch
            {
                ActivationType.Sigmoid =>
                    typeof(TPrecision) == typeof(double) ? (ILayer<TPrecision>)new FullyConnectedSigmoidLayer_64bit(size)
                    ///: typeof(TPrecision) == typeof(float) ? (ILayer<TPrecision>)new FullyConnectedSigmoidLayer_32bit(size)
                    : throw new NotImplementedException(),

                ActivationType.ReLU =>
                    typeof(TPrecision) == typeof(double) ? (ILayer<TPrecision>)new FullyConnectedReLULayer_64bit(size)
                    : throw new NotImplementedException(),

                _ => throw new NotImplementedException()
            };
        }
    }

    public class ConvolutionLayer<TPrecision> : LayerBuilderBase<TPrecision>
    {
        public ConvolutionLayer(ActivationType activation, int inputDepth, int outputDepth, int filterExtend, int stride, int zeroPadding)
        {
            _instance = activation switch
            {
                ActivationType.ReLU =>
                    typeof(TPrecision) == typeof(double) ? (ILayer<TPrecision>)new ConvReLULayer_64bit(inputDepth, outputDepth, filterExtend, stride, zeroPadding)
                    : throw new NotImplementedException(),

                _ => throw new NotImplementedException()
            };
        }
    }
}
