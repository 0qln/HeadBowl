
namespace HeadBowl.Layers
{
    public interface IActivationType
    {
        public IActivation<TPrecision> GetInstance<TPrecision>();
    }

    public interface IActivation<TPrecision>
    {
        public TPrecision Activation(TPrecision value);
        public TPrecision Derivative(TPrecision value);
    }
}
