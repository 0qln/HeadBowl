namespace HeadBowl.TrainingData;

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
