namespace HeadBowl.TrainingData;

public interface ITrainingData<TPrecision>
{
    public int InputSize { get; }
    public int OutputSize { get; }
    public int SampleCount { get; }

    public DataInstance<TPrecision>[] TrainingData { get; }
}
