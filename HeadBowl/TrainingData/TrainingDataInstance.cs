namespace HeadBowl.TrainingData;

public class TrainingDataInstance<T>
{
    public readonly T[] Inputs, Expected;

    public TrainingDataInstance(T[] inputs, T[] expected)
    {
        Inputs = inputs;
        Expected = expected;
    }

    public TrainingDataInstance(int inputSize, int outputSize)
    {
        Inputs = new T[inputSize];
        Expected = new T[outputSize];
    }
}
