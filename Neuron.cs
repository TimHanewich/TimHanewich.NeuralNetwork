using System;
using System.Collections.Generic;
using System.Linq;

namespace TimHanewich.NeuralNetwork
{
    public class Neuron : NeuralComponent
        {
            public float InputValue { get; set; }
            public float IdealValue { get; set; }
            public bool IsBiasNeuron { get; set; }

            //Saving of gamma
            private float LastGamma { get; set; }
            private string LastGammaRequestId { get; set; }
            private float LastValue { get; set; }
            private string LastValueRequestId { get; set; }

            public Dendrite[] GetOutputDendrites(NeuralNetwork ParentNetwork)
            {
                List<Dendrite> MyOutputDend = new List<Dendrite>();
                foreach (Dendrite d in ParentNetwork.Dendrites)
                {
                    if (d.InputNeuronId == Id)
                    {
                        MyOutputDend.Add(d);
                    }
                }
                return MyOutputDend.ToArray();
            }

            public Dendrite[] GetInputDendrites(NeuralNetwork ParentNetwork)
            {
                List<Dendrite> MyInputDend = new List<Dendrite>();
                foreach (Dendrite d in ParentNetwork.Dendrites)
                {
                    if (d.OutputNeuronId == Id)
                    {
                        MyInputDend.Add(d);
                    }
                }
                return MyInputDend.ToArray();
            }

            //The activation function is in here
            public float CalculateValue(NeuralNetwork ParentNetwork, string RequestId = "")
            {
                float ToReturn = 0;
                if (RequestId == "" || RequestId != LastValueRequestId)
                {
                    Dendrite[] MyInputDendrites = GetInputDendrites(ParentNetwork);
                    if (MyInputDendrites.Length == 0) //This is an input neuron!
                    {
                        return InputValue;
                    }
                    else //This is not an input neuron
                    {
                        List<float> WeightedInputs = new List<float>();
                        foreach (Dendrite d in MyInputDendrites)
                        {
                            Neuron ThisDendritesInputNeuron = d.GetInputNeuron(ParentNetwork);
                            WeightedInputs.Add(ThisDendritesInputNeuron.CalculateValue(ParentNetwork) * d.Weight);
                        }

                        //Activation Function (Tanh Function in this case)
                        ToReturn = (float)Math.Tanh(WeightedInputs.Sum());
                        LastValueRequestId = RequestId;
                        LastValue = ToReturn;
                    }
                }
                else
                {
                    ToReturn = LastValue;
                }
                return ToReturn;
            }

            public float CalculateError(NeuralNetwork ParentNetwork)
            {
                float myval = CalculateValue(ParentNetwork);
                float err = 0.5f * (float)Math.Pow(IdealValue - myval, 2);
                return err;
            }

            //The derivative of the activation function is in here
            public float CalculateGamma(NeuralNetwork ParentNetwork, string RequestId = "")
            {
                float ToReturn = 0;
                if (RequestId == "" || RequestId != LastGammaRequestId)
                {
                    float MyVal = CalculateValue(ParentNetwork);
                    float MyValDerivative = (1 - (float)Math.Pow(Math.Tanh(MyVal), 2));

                    Dendrite[] MyOutputDendrites = GetOutputDendrites(ParentNetwork);
                    if (MyOutputDendrites.Length == 0) //I am an output neuron.  This should be simple.
                    {
                        float gamma = (MyVal - IdealValue) * MyValDerivative; //The second part here is the derivative
                        return gamma;
                    }
                    else //I am a hidden (or input) neuron.  I have to reference the gamma of future forward neurons.
                    {
                        List<float> NextLayerWeightedGammas = new List<float>();
                        foreach (Dendrite d in MyOutputDendrites)
                        {
                            NextLayerWeightedGammas.Add(d.Weight * d.GetOutputNeuron(ParentNetwork).CalculateGamma(ParentNetwork));
                        }

                        float gamma = NextLayerWeightedGammas.Sum() * MyValDerivative; //The second part here is the derivative
                        ToReturn = gamma;
                        LastGamma = gamma;
                        LastGammaRequestId = RequestId;
                    }
                }
                else if (RequestId == LastGammaRequestId)
                {
                    ToReturn = LastGamma;
                }

                return ToReturn;
            }
        }
}