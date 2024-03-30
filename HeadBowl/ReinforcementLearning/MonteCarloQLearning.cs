namespace HeadBowl.ReinforcementLearning;

/// <summary>
/// https://github.com/0qln/Obsidian/blob/main/Main/Coding/Q%20Learning.md#monte-carlo-learning
/// </summary>
/// <typeparam name="TPrecision"></typeparam>
/// <typeparam name="TState"></typeparam>
/// <typeparam name="TAction"></typeparam>
public static class MonteCarloQLearning<TPrecision, TState, TAction>
    where TPrecision : struct
{
    /// <summary>
    /// Quality of the State/Action pair.
    /// </summary>
    /// <typeparam name="TState"></typeparam>
    /// <typeparam name="TAction"></typeparam>
    /// <param name="state">Current state.</param>
    /// <param name="action">The action to take.</param>
    /// <returns>The quality of the state s, taken action a.</returns>
    public delegate TPrecision QFunction(TState state, TAction action);

    // TODO:
    // The new function will get slower and slower over each episode of training.
    // Unwrap the recursive implementation and update the function iteratively instead.
    public static QFunction NewQFunction(QFunction oldQFunction, TPrecision totalRewardOverEpisode, int numStates) 
    => (state, action) =>
    {
        TPrecision qOld = oldQFunction(state, action);
        return qOld + (1.0f / numStates) * ((dynamic)totalRewardOverEpisode - qOld);
    };
}