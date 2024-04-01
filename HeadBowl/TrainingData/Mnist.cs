using HeadBowl.Helpers;
using System.Linq;

namespace HeadBowl.TrainingData;

public class Mnist<TPrecision>(string trainImagesPath, string testImagesPath, string trainLabelsPath, string testLabelsPath) : ITrainingData<TPrecision>, ITestingData<TPrecision>
{
    public int InputSize => 28 * 28;
    public int OutputSize => 1;

    private readonly DataInstance<TPrecision>[]
        _trainingData = Mnist<TPrecision>.LoadData(trainImagesPath, trainLabelsPath),
        _testingData = Mnist<TPrecision>.LoadData(testImagesPath, testLabelsPath);

    DataInstance<TPrecision>[] ITestingData<TPrecision>.Data => _testingData;
    DataInstance<TPrecision>[] ITrainingData<TPrecision>.Data => _trainingData;

    int ITrainingData<TPrecision>.SampleCount => _trainingData.Length;
    int ITestingData<TPrecision>.SampleCount => _testingData.Length;

    private static DataInstance<TPrecision>[] LoadData(string imagesPath, string labelsPath) => 
        // Combine Imagees with their coresponding labels.
        Enumerable.Zip(
            // Images
            IDX.Decode<TPrecision>(imagesPath)
               .Cast<Array>()
               .Select(arr => arr.Cast<TPrecision[]>()),
            // Lables
            IDX.Decode<TPrecision>(labelsPath)
               .Cast<TPrecision>())
        // Create a data instace.
        .Select(x => new DataInstance<TPrecision>(
            x.First.SelectMany(y => y).ToArray(),
            [x.Second]))
        .ToArray();
}