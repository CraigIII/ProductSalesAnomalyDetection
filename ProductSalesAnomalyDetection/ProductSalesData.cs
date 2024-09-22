using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ProductSalesAnomalyDetection
{
    internal class ProductSalesData
    {
        [LoadColumn(1)]
        public string Date;

        [LoadColumn(2)]
        public float QP1;
    }
}
