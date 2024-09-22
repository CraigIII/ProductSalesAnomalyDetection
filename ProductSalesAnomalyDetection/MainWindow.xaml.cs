using Microsoft.ML;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;
using OxyPlot.Wpf;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace ProductSalesAnomalyDetection
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private void btnProductSalesAnpmalyDetect_Click(object sender, RoutedEventArgs e)
        {
            MLContext mlContext = new MLContext();

            // 載入資料
            IDataView ProductSales = mlContext.Data.LoadFromTextFile<ProductSalesData>("statsfinal.csv", separatorChar: ',', hasHeader: true);

            // 將訓練資料轉換成List集合
            List<ProductSalesData> data = mlContext.Data.CreateEnumerable<ProductSalesData>(ProductSales, reuseRowObject: false).ToList();
            IDataView dataView = mlContext.Data.LoadFromEnumerable(data);

            //使用線形圖顯示銷售資料
            PlotSalesChart(data);

            Trace.WriteLine("Detect temporary changes in pattern");
            DetectSpike(mlContext, 36, dataView);

            Trace.WriteLine("Detect Persistent changes in pattern");
            DetectChangepoint(mlContext, 36, dataView);
        }
        private void DetectSpike(MLContext mlContext, int docSize, IDataView productSales)
        {
            var iidSpikeEstimator = mlContext.Transforms.DetectIidSpike(outputColumnName: nameof(ProductSalesPrediction.Prediction), inputColumnName: nameof(ProductSalesData.QP1), confidence: 95.0, pvalueHistoryLength: docSize / 4);

            // 訓練
            ITransformer iidSpikeTransform = iidSpikeEstimator.Fit(productSales);

            //取得訓練結果
            IDataView transformedData = iidSpikeTransform.Transform(productSales);
            var predictions = mlContext.Data.CreateEnumerable<ProductSalesPrediction>(transformedData, reuseRowObject: false);

            //顯示預測結果
            Trace.WriteLine("Alert\tScore\tP-Value");
            foreach (var p in predictions)
            {
                var results = $"{p.Prediction[0]}\t{p.Prediction[1]:f2}\t{p.Prediction[2]:F2}";
                if (p.Prediction[0] == 1)
                {
                    results += " <-- Spike detected";
                }
                Trace.WriteLine(results);
            }
        }

        private void DetectChangepoint(MLContext mlContext, int docSize, IDataView productSales)
        {
            var iidChangePointEstimator = mlContext.Transforms.DetectIidChangePoint(outputColumnName: nameof(ProductSalesPrediction.Prediction), inputColumnName: nameof(ProductSalesData.QP1), confidence: 95.0, changeHistoryLength: docSize / 4);

            //訓練
            var iidChangePointTransform = iidChangePointEstimator.Fit(productSales);

            //取得訓練結果
            IDataView transformedData = iidChangePointTransform.Transform(productSales);
            var predictions = mlContext.Data.CreateEnumerable<ProductSalesPrediction>(transformedData, reuseRowObject: false);

            //顯示預測結果
            Trace.WriteLine("Alert\tScore\tP-Value\tMartingale value");
            foreach (var p in predictions)
            {
                var results = $"{p.Prediction[0]}\t{p.Prediction[1]:f2}\t{p.Prediction[2]:F2}\t{p.Prediction[3]:F2}";
                if (p.Prediction[0] == 1)
                {
                    results += " <-- alert is on, predicted changepoint";
                }
                Trace.WriteLine(results);
            }
        }

        private void PlotSalesChart(List<ProductSalesData> productSales)
        {
            //建立繪圖模型
            var plotModel = new PlotModel { Title = "Sales Data" };
            //建立線形圖需要的序列資料
            var lineSeries = new LineSeries
            {
                Title = "Sales",
                MarkerType = MarkerType.Circle,
                ItemsSource = productSales,
                Mapping = item =>
                {
                    var data = (ProductSalesData)item;
                    DateTime date;
                    if (DateTime.TryParseExact(data.Date, "dd-MM-yyyy", null, System.Globalization.DateTimeStyles.None, out date))
                    {
                        return new DataPoint(DateTimeAxis.ToDouble(date), data.QP1);
                    }
                    else
                    {
                        throw new FormatException($"The input string '{data.Date}' was not in a correct format.");
                    }
                }
            };
            //將序列資料加入到繪圖模型
            plotModel.Series.Add(lineSeries);
            //建立顯示繪圖模型的PlotView檢視
            var plotView = new PlotView
            {
                Model = plotModel,
            };
            //顯示PlotView檢視
            Content = plotView;
        }
    }
}