﻿using System.Diagnostics;
using System.Text;
using HeadBowl.ReinforcementLearning;

using Bitboard = int;
using Color = int;
using GameState = int;

namespace HeadBowl.TrainingData.TicTacToe;

public struct Position : ITwoAgentEnviroment<int>
{
    /// <summary>
    /// Indexed by color
    /// </summary>
    private readonly Bitboard[] _occupactions = new Bitboard[3];

    /// <summary>
    /// A bitboard of the occupied squares.
    /// </summary>
    public readonly Bitboard Occupation => _occupactions[Colors.O] | _occupactions[Colors.X];

    /// <summary>
    /// Indexed by square
    /// </summary>
    private readonly Color[] _colors = new Color[9];

    public readonly Color[] SquareColors => _colors;

    /// <summary>
    /// The current color at the square.
    /// </summary>
    /// <param name="square"></param>
    /// <returns></returns>
    public readonly Color this[int square]
    {
        get
        {
            return _colors[square];
        }
    }

    /// <summary>
    /// The current square at the position.
    /// </summary>
    /// <param name="row"></param>
    /// <param name="col"></param>
    /// <returns></returns>
    public readonly Color this[int row, int col]
    {
        get
        {
            return this[row * 3 + col];
        }
    }

    /// <summary>
    /// The current game state.
    /// </summary>
    public GameState GameState { get; private set; } = GameStates.Ongoing;

    /// <summary>
    /// Wether the position is terminal or not.
    /// </summary>
    public readonly bool Terminal => GameState != GameStates.Ongoing;
    
    /// <summary>
    /// Relative to the last player to make a move.
    /// </summary>
    public readonly int Reward => Terminal ? (GameState == GameStates.Draw ? Rewards.Draw : Rewards.Win) : Rewards.None;

    /// <summary>
    /// The side to move.
    /// </summary>
    public Color Turn { get; private set; } = Colors.X;
    public readonly Color Agent1 => Colors.X;
    public readonly Color Agent2 => Colors.O;

    /// <summary>
    /// Default constructor.
    /// </summary>
    public Position()
    {
    }


    public readonly void RemoveColor(int square)
    {
        _occupactions[_colors[square]] ^= 1 << square;
        _colors[square] = Colors.None;
    }

    public readonly void AddColor(int square, Color color)
    {
        _occupactions[color] |= 1 << square;
        _colors[square] = color;
    }

    /// <summary>
    /// Make a move.
    /// </summary>
    /// <param name="square"></param>
    public void MakeAction(int square)
    {
        Debug.Assert(GameState == GameStates.Ongoing);

        // Add the color.
        AddColor(square, Turn);

        // Update game state.
        for (int ray = 0; ray < 4; ray++)
        {
            Bitboard bb = Utils.Rays[square, ray];
            if (bb != 0 && (bb & _occupactions[Turn]) == bb)
            {
                GameState = GameStates.Win[Turn];
                break;
            }
        }

        if (GameState == GameStates.Ongoing && Occupation == 0b_111_111_111)
            GameState = GameStates.Draw;

        // Chagne turn.
        Turn = Turn.Inv();
    }

    /// <summary>
    /// Undo a move.
    /// </summary>
    /// <param name="square"></param>
    public void UndoAction(int square)
    {
        // Add the color.
        RemoveColor(square);

        // Chagne turn.
        Turn = Turn.Inv();

        // Update game state.
        GameState = GameStates.Ongoing;
    }

    /// <summary>
    /// Get an Iterator of moves that can be made in this position.
    /// </summary>
    /// <returns></returns>
    public readonly IEnumerable<int> LegalActions()
    {
        if (GameState == GameStates.Ongoing)
        {
            Bitboard bb = Occupation ^ 0b_111_111_111;
            while (bb != 0)
                yield return Utils.PopLsb(ref bb);
        }
    }

    public readonly void RemoveColor(int row, int col) => RemoveColor(row * 3 + col);

    public readonly void AddColor(int row, int col, Color color) => AddColor(row * 3 + col, color);

    public readonly override string? ToString()
    {
        StringBuilder sb = new("+---+---+---+\n");
        for (int row = 2; row >= 0; row--)
        {
            for (int col = 0; col < 3; col++)
                sb.Append($"| {Colors.ToString(this[row, col])} ");
            sb.AppendLine("|");
            sb.AppendLine("+---+---+---+");
        }
        return sb.ToString();
    }


    ulong Perft()
    {
        ulong result = 0;

        foreach (var move in LegalActions())
        {
            MakeAction(move);
            result += Perft();
            UndoAction(move);
        }

        return result == 0 ? 1 : result;
    }
}