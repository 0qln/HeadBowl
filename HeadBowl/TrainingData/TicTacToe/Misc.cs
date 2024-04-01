using System.Numerics;

using Bitboard = int;
using Color = int;
using GameState = int;

namespace HeadBowl.TrainingData.TicTacToe;

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