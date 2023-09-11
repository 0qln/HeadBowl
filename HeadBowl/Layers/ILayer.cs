﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HeadBowl.Layers
{
    public interface ILayer<TFloat> : IInitializable
    {
        public int Size { get; }
        public TFloat[] Values { get; }
    }
}
