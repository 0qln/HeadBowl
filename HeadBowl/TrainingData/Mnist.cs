namespace HeadBowl.TrainingData;

public class Mnist<TPrecision> : ITrainingData<TPrecision>, ITestingData<TPrecision>
{
    private readonly DataInstance<TPrecision>[] _trainingData, _testingData;

    public int InputSize => throw new NotImplementedException();
    public int OutputSize => throw new NotImplementedException();
    public int SampleCount => throw new NotImplementedException();

    DataInstance<TPrecision>[] ITestingData<TPrecision>.Data => _testingData;
    DataInstance<TPrecision>[] ITrainingData<TPrecision>.Data => _trainingData;


    public Mnist(string trainImages, string testImages, string trainLabels, string testLabels)
    {
        throw new NotImplementedException();
    }
}