namespace HeadBowl.ReinforcementLearning;

public interface IEnviroment<TAction>
{
    public bool Terminal { get; }

    public IEnumerable<TAction> LegalActions();

    public void MakeAction(TAction action);

    public void UndoAction(TAction action);

    /// <summary>
    /// Relative to the last player to make a move.
    /// </summary>
    public int Reward { get; }
}