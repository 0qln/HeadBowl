using System.Diagnostics;
using System.Numerics;

namespace HeadBowl.TrainingData.Enviroments;

using System.Text;
using Bitboard = int;
using Color = int;
using GameState = int;

public static class GameStates
{
    public const GameState
        Ongoing = -1,
        Draw = 0,
        WinX = Colors.X,
        WinO = Colors.O;

    public static readonly GameState[] Win;

    static GameStates()
    {
        Win = new GameState[3];
        Win[Colors.O] = GameStates.WinO;
        Win[Colors.X] = GameStates.WinX;
    }
}

public static class Colors
{
    public const Color
        None = 0,
        X = 1,
        O = 2;

    public static string ToString(Color color) => color == X ? "X" : color == O ? "O" : " ";

    public static Color Inv(this Color color) => color == X ? O : X;
}

public static class Utils
{
    public static readonly Bitboard[,] Rays;

    static Utils()
    {
        Rays = new Bitboard[9, 4];
        for (int square = 0; square < 9; square++)
        {
            int row = square / 3, col = square % 3;
            Rays[square, 0] = Rows[row];
            Rays[square, 1] = Cols[col];

            // Diagonals
            Rays[square, 2] =
                (square == 0 || square == 4 || square == 8)
                ? 0b_100_010_001
                : 0b_0;

            Rays[square, 3] =
                (square == 2 || square == 4 || square == 6)
                ? 0b_001_010_100
                : 0b_0;
        }
    }

    public static readonly Bitboard[] Rows =
    [
        0b111 << 0,
        0b111 << 3,
        0b111 << 6,
    ];

    public static readonly Bitboard[] Cols =
    [
        0b001001001,
        0b010010010,
        0b100100100,
    ];

    public static int PopLsb(ref Bitboard board)
    {
        unchecked
        {
            int lsb = BitOperations.TrailingZeroCount(board);
            board &= board - 1;
            return lsb;
        }
    }

    public static string SqToStr(int square)
    {
        int row = square / 3, col = square % 3;
        return ((char)(col + 'a')).ToString() + ((char)(row + '1')).ToString();
    }

    public static int StrToSq(string square)
    {
        int row = square[1] - '1', col = square[0] - 'a';
        return row * 3 + col;
    }
}

public struct Position
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
    /// The side to move.
    /// </summary>
    public Color Turn { get; private set; } = Colors.X;


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
    public void MakeMove(int square)
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
    public void UndoMove(int square)
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
    public readonly IEnumerable<int> LegalMoves()
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

        foreach (var move in LegalMoves())
        {
            MakeMove(move);
            result += Perft();
            UndoMove(move);
        }

        return result == 0 ? 1 : result;
    }
}