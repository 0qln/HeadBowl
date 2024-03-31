namespace HeadBowl.TrainingData;

public class DataInstance<TPrecision>
{
    public readonly TPrecision[] Inputs, Expected;

    public DataInstance(TPrecision[] inputs, TPrecision[] expected)
    {
        Inputs = inputs;
        Expected = expected;
    }

    public DataInstance(int inputSize, int outputSize)
    {
        Inputs = new TPrecision[inputSize];
        Expected = new TPrecision[outputSize];
    }
}
