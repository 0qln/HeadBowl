
using HeadBowl.Nets;
using HeadBowl.ReinforcementLearning;
using Bitboard = int;
using Color = int;
using GameState = int;

namespace HeadBowl.TrainingData.TicTacToe;

public class Engine<TPrecision> : IAgent<TPrecision, Position, int>
    where TPrecision : struct
{
    private INet<TPrecision> _nn;

    public Engine(INet<TPrecision> nn)
    {
        _nn = nn;
    }

    private TPrecision[] GetInputs(Position state, int action)
    {
        var inputs = new TPrecision[state.SquareColors.Length + 1];
        Array.Copy(state.SquareColors, inputs, state.SquareColors.Length);
        inputs[state.SquareColors.Length] = (TPrecision)(dynamic)action;
        return inputs;
    }

    Functions.QFunction<TPrecision, Position, int> IAgent<TPrecision, Position, int>.Q => 
        (env, action) =>
    {
        var inputs = GetInputs(env, action);
        var outputs = _nn.Forward(inputs);
        return outputs[0];
    };
}