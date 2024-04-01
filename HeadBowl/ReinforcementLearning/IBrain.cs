using HeadBowl.Nets;
using HeadBowl.TrainingData.TicTacToe;

namespace HeadBowl.ReinforcementLearning;

public interface IBrain<TPrecision>
{
    public INet<TPrecision> Net { get; }
}