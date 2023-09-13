using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HeadBowl.OLD.Activations
{
    public interface IActivationFunction<TFloat>
    {
        /// <summary>
        /// Normal function
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public TFloat Forward(TFloat value);

        /// <summary>
        /// Derivative
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public TFloat Backward(TFloat value);
    }
}
