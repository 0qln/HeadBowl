namespace HeadBowl.ReinforcementLearning;

public static class Functions
{
    /// <summary>
    /// Quality of the State/Action pair.
    /// </summary>
    /// <typeparam name="TState">The type of the state.</typeparam>
    /// <typeparam name="TAction">The type of the action.</typeparam>
    /// <param name="state">Current state.</param>
    /// <param name="action">The action to take.</param>
    /// <returns>The quality of the state s, taken action a.</returns>
    public delegate TPrecision QFunction<TPrecision, TState, TAction>(TState state, TAction action)
        where TPrecision : struct;
}