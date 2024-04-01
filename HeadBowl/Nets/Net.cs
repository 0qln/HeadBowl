using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using HeadBowl.Layers;
using HeadBowl.Optimizers;
using Microsoft.Toolkit.HighPerformance;
using Microsoft.Win32.SafeHandles;

namespace HeadBowl.Nets
{
    public interface INet<T>
    {
        public T Cost { get; }
        /// <summary>
        /// Recommendet for Nets with large layer sizes, but few layer count. 
        /// </summary>
        public bool EnableParallelProcessing { get; set; }
        public bool ExperimentalFeature { get; set; }
        public void Train(T[] inputs, T[] expectedOutputs);
        public T[] Forward(Array inputs);

        public ILayer<T>[] Layers { get; }

        public string Safe(string folder);
        public void Load(string file, bool destroyAfterRead = false);
    }



    public static class Net
    {
        public static INet<TPrecision> Build<TPrecision>(params ILayerBuilder<TPrecision>[] layerBuilders)
        {
            var layers = new List<ILayer<TPrecision>>();

            layerBuilders[0].SetNext(layerBuilders[1].Instance());
            layerBuilders[^1].SetPrev(layerBuilders[^2].Instance());
            for (int layer = 1; layer < layerBuilders.Length - 1; layer++)
            {
                layerBuilders[layer].SetPrev(layerBuilders[layer - 1].Instance());
                layerBuilders[layer].SetNext(layerBuilders[layer + 1].Instance());
            }

            foreach (var layer in layerBuilders)
            {
                layers.Add(layer.Build());
            }

            return 
                typeof(TPrecision) == typeof(Half) ? throw new NotImplementedException() :
                typeof(TPrecision) == typeof(float) ? throw new NotImplementedException() :
                typeof(TPrecision) == typeof(double) ? (INet<TPrecision>)new Net_64bit((ILayer<double>[])layers.ToArray()) :
                typeof(TPrecision) == typeof(decimal) ? throw new NotImplementedException()
                : throw new NotImplementedException();
        }

        public static INet<TPrecision> Clone<TPrecision>(INet<TPrecision> original)
        {
            var layerBuilders = new List<ILayerBuilder<TPrecision>>();
            for (int i = 0; i < original.Layers.Length; i++)
            {
                layerBuilders.Add(original.Layers[i].ToRawBuilder());
            }
            var copy = Build(layerBuilders.ToArray());

            for (int i = 0; i < copy.Layers.Length; i++)
            {
                copy.Layers[i].Weights = (TPrecision[,])original.Layers[i].Weights.Clone();
                copy.Layers[i].Biases = (TPrecision[])original.Layers[i].Biases.Clone();
            }

            return copy;
        }
        public static void SetOptimizer<TPrecision>(INet<TPrecision> net, Func<IOptimizer<TPrecision>> optimizer)
        {
            foreach (var layer in net.Layers)
            {
                layer.Optimizer = optimizer();
            }
        }
    }


    internal class Net_64bit : INet<double>
    {
        public Array? Outputs => _layers[^1].Activations;
        public double Cost => _lastCost;
        public bool EnableParallelProcessing
        {
            get
            {
                return _enableParallelProcessing;
            }
            set
            {
                _enableParallelProcessing = value;
                foreach (var layer in _layers)
                {
                    layer.EnableParallelProcessing = value;
                }
            }
        }
        public bool ExperimentalFeature
        {
            get
            {
                return _experimentalFeature;
            }
            set
            {
                _experimentalFeature = value;
                foreach (var layer in _layers)
                {
                    layer.ExperimentalFeature = value;
                }
            }
        }

        private bool _experimentalFeature = false;
        private bool _enableParallelProcessing = false;
        private ILayer<double>[] _layers;
        private double _lastCost = 0;

        public ILayer<double>[] Layers => _layers;


        public Net_64bit(params ILayer<double>[] layers)
        {
            _layers = layers;
        }

        public static void MSE(double[] outputs, double[] expected, out double result)
        {
            result = 0;
            for (int i = 0; i < outputs.Length; i++)
                result += Math.Pow(outputs[i] - expected[i], 2);
            result /= expected.Length;
        }

        public void Train(double[] inputs, double[] expectedOutputs)
        {
            Forward(inputs);

            MSE((double[])_layers[^1].Activations! ?? throw new Exception("Last layer has to have an onedimensional array as output. Consider rethinking your network design."),
                expectedOutputs,
                out _lastCost);

            _layers.Last().GradientDependencies = expectedOutputs;

            foreach (var layer in _layers.Reverse())
            {
                layer.GenerateGradients();
                layer.ApplyOptimizer();
                layer.ApplyGradients();
                layer.UpdateParamaters();
            }
        }

        public double[] Forward(Array inputs)
        {
            _layers[0].Activations = inputs;

            for (int layer = 1; layer < _layers.Length; layer++)
            {
                _layers[layer].Forward();
            }

            return (double[])_layers[^1].Activations!
                ?? throw new Exception("Last layer has to have an onedimensional array as output. Consider rethinking your network design.");
        }

        public override string? ToString()
        {
            StringBuilder sb = new();
            sb.Append("net");
            sb.Append("-64bit");
            sb.Append("-layers");
            foreach (var layer in _layers)
                sb.Append($"_{layer.Size}");
            return sb.ToString();
        }

        string INet<double>.Safe(string folder)
        {
            int fileNum = 0;
            string name;
            string destination() => Path.Combine(folder, name) + ".dat";
            
            // Find a valid name.
            do
                name = $"{this}-backup-({fileNum++})";
            while (File.Exists(destination()));

            // Create the file.
            if (!Directory.Exists(folder))
                Directory.CreateDirectory(folder);
            using BinaryWriter fs = new(File.Open(destination(), FileMode.Create));
            foreach (var layer in _layers)
            {
                for (int i = 0; i < layer.Weights.GetLength(0); i++)
                    for (int j = 0; j < layer.Weights.GetLength(1); j++)
                        fs.Write(((double[,])layer.Weights)[i, j]);

                for (int i = 0; i < layer.Biases.Length; i++)
                    fs.Write(((double[])layer.Biases)[i]);
            }

            return destination();
        }

        // eg. "net-64bit-layers_10_200_200_1-backup-(0)"
        void INet<double>.Load(string file, bool destroyAfterRead)
        {
            // Check bounds and hyper parameters
            if (Regex.Match(file, @"(?<=net-).*(?=bit-)").Value != "64")
            {
                throw new ArgumentException("Invalid precision.");
            }

            foreach (var x in _layers.Zip(Regex.Match(file, @"(?<=-layers_)[\d_]+").Value.Split('_')))
            {
                if (x.First.Size != int.Parse(x.Second))
                    throw new ArgumentException("Invalid layer sizes.");
            }

            // Load parameters
            using (BinaryReader fs = new(File.OpenRead(file)))
            {
                foreach (var layer in _layers)
                {
                    for (int i = 0; i < layer.Weights.GetLength(0); i++)
                        for (int j = 0; j < layer.Weights.GetLength(1); j++)
                            ((double[,])layer.Weights)[i, j] = fs.ReadDouble();

                    for (int i = 0; i < layer.Biases.Length; i++)
                        ((double[])layer.Biases)[i] = fs.ReadDouble();
                }
            }

            if (destroyAfterRead)
            {
                File.Delete(file);
            }
        }
    }
}
