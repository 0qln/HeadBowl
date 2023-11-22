using HeadBowl.Helpers;
using HeadBowl.Optimizers;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace HeadBowl.Layers
{
    public class Sigmoid : IActivationType
    {
        public IActivation<TPrecision> GetInstance<TPrecision>()
        {
            return
                typeof(TPrecision) == typeof(double) ? (IActivation<TPrecision>)new Sigmoid_64bit() :
                typeof(TPrecision) == typeof(float) ? (IActivation<TPrecision>)new Sigmoid_32bit() :
                throw new NotImplementedException();
        }
    }

    internal interface ISigmoid<T> : IActivation<T>
    {
        /// <summary>When this is used, the input should be already sigmoided.</summary>
        public new T Derivative(T input);
    }

    internal class Sigmoid_64bit : ISigmoid<double>
    {
        public double Activation(double input) => 1 / (1 + Math.Exp(-input));
        public double Derivative(double input) => input * (1 - input); 
    }

    internal class Sigmoid_32bit : ISigmoid<float>
    {
        public float Activation(float input) => 1f / (float)(1 + Math.Exp(-input));
        public float Derivative(float input) => input * (1f - input);
    }
}
