namespace HeadBowl.Helpers
{
    internal static class Init<T>
    {
        private static Random rng = new();

        public static T[,] Weights(int size1, int size2)
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

        public static T[] Biases(int size)
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
                result[i] = (dynamic)0.00001;
            }

            return result;
        }
    }

}
