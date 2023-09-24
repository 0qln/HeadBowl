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

}
