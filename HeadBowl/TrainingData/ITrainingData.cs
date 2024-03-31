namespace HeadBowl.TrainingData;

public interface ITrainingData<T>
{
    public int InputSize { get; }
    public int OutputSize { get; }
    public int SampleCount { get; }

    public TrainingDataInstance<T>[] Data { get; }
}
