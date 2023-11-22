
using System.Diagnostics;

namespace HeadBowl.Helpers
{
    internal static class Init<T>
    {
        private static Random rng = new();

        public static T[,,,] Random(params int[] sizes)
        {
            Debug.Assert(sizes.Length == 4);

            var result = new T[
                sizes[0],
                sizes[1],
                sizes[2],
                sizes[3]
            ];

            for (int i = 0; i < sizes[0];  i++)            
                for (int j = 0; j < sizes[0]; j++)                
                    for (int k = 0; k < sizes[0]; k++)                    
                        for (int l = 0; l < sizes[0]; l++)                        
                            result[i,j,k,l] = (dynamic)rng.NextDouble();
                        

            return result;
        }

        public static T[,] Random(int size1, int size2)
        {
            var result = new T[size1, size2];

            for (int i = 0; i < size1; i++)
            {
                for (int j = 0; j < size2; j++)
                {
                    result[i, j] = (dynamic)rng.NextDouble();
                }
            }

            return result;
        }

        public static T[] Random(int size)
        {
            var result = new T[size];

            for (int i = 0; i < size; i++)
            {
                result[i] = (dynamic)rng.NextDouble();
            }

            return result;
        }

        public static T[] LearningRates(int size)
        {
            var result = new T[size];

            for (int i = 0; i < size; i++)
            {
                result[i] = (dynamic)0.01;
            }

            return result;
        }
    }

}
