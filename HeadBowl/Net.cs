using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HeadBowl
{
    internal interface INet<T>
    {

    }



    public class Net<TPrecision> : INet<TPrecision>
    {
        private INet<TPrecision> _net;


        public Net(params ILayer<TPrecision>[] layers)
        {            
            _net =
                typeof(TPrecision) == typeof(Half) ? throw new NotImplementedException() :
                typeof(TPrecision) == typeof(float) ? throw new NotImplementedException() :
                typeof(TPrecision) == typeof(double) ? (INet<TPrecision>)new Net_64bit((ILayer<double>[])layers) :
                typeof(TPrecision) == typeof(decimal) ? throw new NotImplementedException() 
                : throw new NotImplementedException();
        }
    }


    internal class Net_64bit : INet<double>
    {
        public double[] Outputs => _layers[^1].Activations;
        
        private ILayer<double>[] _layers;


        
        public Net_64bit(params ILayer<double>[] layers)
        {
            _layers = layers;

        }


        public void Forward(double[] inputs)
        {

        }

        public void Train(double[] inputs, double[] expectedOutputs)
        {

        }
    }
}
