using HeadBowl.Activations;
using HeadBowl.Layers;
using HeadBowl.Nets;
using HeadBowl.Training_Data;


public static class Program
{
    public static void Main(string[] args)
    {
        var net = new Net<float>(new InputLayer<float>(2),
            new LayerBuilder< ReLU<float>, float>(10),
            new LayerBuilder< ReLU<float>, float>(1));


        net.Forward(Xor<float>.Data[0].Inputs);

        for (int i = 0;  i < 10000; i++)
        {
            foreach (var data in Xor<float>.Data)
            {
                net.Backward(data.Inputs, data.Expected);
            }
        }

        net.Forward(Xor<float>.Data[0].Inputs);
    }
}