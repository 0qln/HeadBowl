using System.Buffers.Binary;
using System.Text.RegularExpressions;

namespace HeadBowl.Helpers;

/// <summary>
/// https://www.fon.hum.uva.nl/praat/manual/IDX_file_format.html
/// </summary>
public static class IDX
{
    public readonly struct Magic
    {
        public const int IDX_DATATYPE = 2;
        public const int IDX_DIMENSIONS = 3;

        public readonly byte Dimensions;
        public readonly Type DataType;
        public readonly Func<BinaryReader, object> NextData;

        public Magic(uint raw)
        {
            var bytes = BitConverter.GetBytes(raw);
            
            Dimensions = bytes[IDX_DIMENSIONS];

            DataType = bytes[IDX_DATATYPE] switch
            {
                0x08 => typeof(byte),
                0x09 => typeof(sbyte),
                0x0B => typeof(short),
                0x0C => typeof(int),
                0x0D => typeof(float),
                0x0E => typeof(double),
                _ => throw new ArgumentException("Invalid magic number.")
            };

            // TODO: The IDX documentation is unclear, wether the data is stored in little or big endian encoding.
            NextData = bytes[IDX_DATATYPE] switch
            {
                0x08 => reader => reader.ReadByte(),
                0x09 => reader => reader.ReadSByte(),
                0x0B => reader => reader.ReadInt16(),
                0x0C => reader => reader.ReadInt32(),
                0x0D => reader => reader.ReadSingle(),
                0x0E => reader => reader.ReadDouble(),
                _ => throw new ArgumentException("Invalid magic number.")
            };
        }
    }

    public static Array Decode<TPrecision>(string file)
    {
        using BinaryReader reader = new(File.OpenRead(file));

        if (Path.GetExtension(file)[..4] != ".idx")
        {
            throw new ArgumentException("Invalid file extension.");
        }

        Magic magic = new(reader.ReadUInt32());
        int[] sizes = new int[magic.Dimensions];
        for (int i = 0; i < magic.Dimensions; i++)
            sizes[i] = BinaryPrimitives.ReadInt32BigEndian(reader.ReadBytes(4));

        return Read(sizes, reader, ref magic);

        // Recursive function to deal with unknown dimension count.
        static Array Read(int[] sizes, BinaryReader reader, ref Magic magic)
        {
            int dimensions = sizes.Length;
            int size = sizes[0];

            if (dimensions == 1)
            {
                TPrecision[] result = new TPrecision[size];
                for (int i = 0; i < size; ++i)
                {
                    result[i] = (TPrecision)(dynamic)magic.NextData(reader);
                }
                return result;
            }
            else
            {
                Array[] result = new Array[size];
                for (int i = 0; i < size; ++i)
                {
                    result[i] = Read(sizes[1..], reader, ref magic);
                }
                return result;
            }
        }
    }
}
