using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HeadBowl.Training_Data
{
    public static class Xor<TFloat>
    {
        public static readonly (TFloat[] Inputs, TFloat[] Expected)[] Data = new (TFloat[], TFloat[])[] {
            (new TFloat[]{ (dynamic)0, (dynamic)0 }, new TFloat[]{ (dynamic)0 } ),
            (new TFloat[]{ (dynamic)1, (dynamic)0 }, new TFloat[]{ (dynamic)1 } ),
            (new TFloat[]{ (dynamic)0, (dynamic)1 }, new TFloat[]{ (dynamic)1 } ),
            (new TFloat[]{ (dynamic)1, (dynamic)1 }, new TFloat[]{ (dynamic)0 } ),
        };


    }
}
