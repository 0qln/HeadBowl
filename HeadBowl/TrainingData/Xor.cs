namespace HeadBowl.TrainingData;

public interface ITrainingData<T>
{
    public int InputSize { get; }
    public int OutputSize { get; }
    public int SampleCount { get; }

    public TrainingDataInstance<T>[] Data { get; }
}

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

public class Xor<T> : ITrainingData<T>
{
    public int InputSize { get; } = 2;
    public int OutputSize { get; } = 1;
    public int SampleCount => Data.Length;

    public TrainingDataInstance<T>[] Data =>
    [
        new([(dynamic)0, (dynamic)0], [(dynamic)0]),
        new([(dynamic)1, (dynamic)0], [(dynamic)1]),
        new([(dynamic)0, (dynamic)1], [(dynamic)1]),
        new([(dynamic)1, (dynamic)1], [(dynamic)0]),
    ];
}
