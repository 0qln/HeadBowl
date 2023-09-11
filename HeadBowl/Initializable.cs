using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HeadBowl
{
    /// <summary>
    /// Implementations of this class should have an empty, parameterless constructor aswell.
    /// </summary>
    public interface IInitializable
    {
        public IInitializable Init(params object[] parameters);
    }
}
