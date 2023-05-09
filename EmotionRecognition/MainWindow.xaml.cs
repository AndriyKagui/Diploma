using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media.Imaging;
using OpenCvSharp;
using Window = System.Windows.Window;
using System;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using System.Collections.Generic;
using System.Linq;
using Size = OpenCvSharp.Size;
using Rect = OpenCvSharp.Rect;
using Point = OpenCvSharp.Point;

namespace EmotionRecognition
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private VideoCapture _capture;
        private const string faceCascadePath = "haarcascade_frontalface_default.xml";
        private const string modelPath = "emotion_detection_model.onnx";
        private string[] classLabels = new[] { "Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprised" };


        public MainWindow()
        {
            InitializeComponent();
            LoadAvailableCameras();
        }

        private async void StartButton_Click(object sender, RoutedEventArgs e)
        {
            if (_capture != null)
            {
                _capture.Release();
            }

            int selectedIndex = videoSourceComboBox.SelectedIndex;
            if (selectedIndex < 0)
            {
                MessageBox.Show("No camera selected.", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            _capture = new VideoCapture(selectedIndex);
            if (!_capture.IsOpened())
            {
                MessageBox.Show("Failed to open the selected camera.", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            while (_capture.IsOpened())
            {
                using Mat frame = new Mat();
                _capture.Read(frame);

                if (frame.Empty())
                {
                    break;
                }

                var faceClassifier = new CascadeClassifier(faceCascadePath);
                var classifier = new InferenceSession(modelPath);

                var gray = frame.CvtColor(ColorConversionCodes.BGR2GRAY);
                var faces = faceClassifier.DetectMultiScale(gray, 1.2, 4);

                foreach (var face in faces)
                {
                    frame.Rectangle(face, Scalar.Blue, 2);
                    var roiGray = new Rect(face.X, face.Y, face.Width, face.Height);

                    var preprocessedFrame = Preprocess(gray[roiGray]);
                    var output = RunOnnxModel(preprocessedFrame, classifier);

                    var maxIndex = Array.IndexOf(output, output.Max());
                    var label = classLabels[maxIndex];
                    var labelPosition = new Point(face.X, face.Y);

                    frame.PutText(label, labelPosition, HersheyFonts.HersheySimplex, 2, Scalar.Green, 3);
                    emotionLabel.Text = label;
                }

                // Конвертування зображення OpenCV до BitmapSource для відображення у WPF
                var bitmap = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(frame);
                var source = System.Windows.Interop.Imaging.CreateBitmapSourceFromHBitmap(bitmap.GetHbitmap(), IntPtr.Zero, Int32Rect.Empty, BitmapSizeOptions.FromEmptyOptions());
                cameraImage.Source = source;

                // Дозволити обробку подій та інші завдання у UI
                await Task.Delay(1);
            }
        }


        private void StopButton_Click(object sender, RoutedEventArgs e)
        {
            if (_capture != null)
            {
                _capture.Release();
            }
            cameraImage.Source = null;
        }


        private void LoadAvailableCameras()
        {
            for (int i = 0; i < 5; i++)
            {
                using var capture = new VideoCapture(i);
                if (capture.IsOpened())
                {
                    videoSourceComboBox.Items.Add($"Camera {i}");
                }
            }

            if (videoSourceComboBox.Items.Count > 0)
            {
                videoSourceComboBox.SelectedIndex = 0;
            }
        }

        private static Tensor<float> Preprocess(Mat frame)
        {
            Cv2.Resize(frame, frame, new Size(48, 48));
            frame.ConvertTo(frame, MatType.CV_32F);
            frame /= 255.0f;

            var inputData = new float[1 * 48 * 48 * 1];
            int index = 0;
            for (int i = 0; i < 48; i++)
            {
                for (int j = 0; j < 48; j++)
                {
                    inputData[index++] = frame.At<float>(i, j);
                }
            }

            var tensor = new DenseTensor<float>(inputData, new[] { 1, 48, 48, 1 });
            return tensor;
        }

        private static float[] RunOnnxModel(Tensor<float> input, InferenceSession session)
        {
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("conv2d_1_input", input) };
            using var results = session.Run(inputs);
            var output = results.First().AsEnumerable<float>().ToArray();
            return output;
        }
    }
}
