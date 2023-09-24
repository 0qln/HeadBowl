using HeadBowl;
using HeadBowl.Training_Data;


public static class Program
{
    public static void Main(string[] args)
    {
        ITrainingData<double> traningData = new Xor<double>();

        INet<double> nn = Net<double>.Build(
            new ReLULayerBuilder<double>(2),
            new ReLULayerBuilder<double>(50),
            new ReLULayerBuilder<double>(1));


        for (int i = 0;  i < 100000; i++)
        {
            foreach (var data in traningData.Data)
            {
                nn.Train(data.Inputs, data.Expected);
            }

            if (i % 1000 == 0)
                Console.WriteLine(nn.Cost);
        }
    }
}