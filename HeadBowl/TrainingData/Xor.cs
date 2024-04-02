namespace HeadBowl.TrainingData;

public class Xor<TPrecision> : ITrainingData<TPrecision>
{
    public int InputSize { get; } = 2;
    public int OutputSize { get; } = 1;
    public int SampleCount => (this as ITrainingData<TPrecision>).TrainingData.Length;

    DataInstance<TPrecision>[] ITrainingData<TPrecision>.TrainingData =>
    [
        new([(dynamic)0, (dynamic)0], [(dynamic)0]),
        new([(dynamic)1, (dynamic)0], [(dynamic)1]),
        new([(dynamic)0, (dynamic)1], [(dynamic)1]),
        new([(dynamic)1, (dynamic)1], [(dynamic)0]),
    ];
}
