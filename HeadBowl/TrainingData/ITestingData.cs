namespace HeadBowl.TrainingData;

public interface ITestingData<TPrecision>
{
    public int InputSize { get; }
    public int OutputSize { get; }
    public int SampleCount { get; }

    public DataInstance<TPrecision>[] TestingData { get; }
}