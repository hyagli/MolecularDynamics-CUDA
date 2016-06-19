using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.IO;
using System.Text.RegularExpressions;
using System.Globalization;
using System.Windows.Forms.DataVisualization.Charting;

namespace mdGraph
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();

            if (Properties.Settings.Default.WindowMaximized == true)
                WindowState = FormWindowState.Maximized;

            chart1.MouseWheel += new MouseEventHandler(chart_MouseWheel);

            string[] args = Environment.GetCommandLineArgs();
            if (args.Length > 1)
            {
                LoadPointsFromFile(args[1]);
            }
        }

        private void chart_MouseWheel(object sender, MouseEventArgs e)
        {
            ChartArea area = chart1.ChartAreas[0];
            double xValue = area.AxisX.PixelPositionToValue(e.X);
            double yValue = area.AxisY.PixelPositionToValue(e.Y);
            double newXsize = 1;
            double newYsize = 1;
            
            if (e.Delta < 0)
            {                
                newXsize = (area.AxisX.ScaleView.ViewMaximum - area.AxisX.ScaleView.ViewMinimum) / 2;
                newYsize = (area.AxisY.ScaleView.ViewMaximum - area.AxisY.ScaleView.ViewMinimum) / 2;
            }
            else
            {
                newXsize = (area.AxisX.ScaleView.ViewMaximum - area.AxisX.ScaleView.ViewMinimum) * 2;
                newYsize = (area.AxisY.ScaleView.ViewMaximum - area.AxisY.ScaleView.ViewMinimum) * 2;
            }

            double newXstart = Math.Round(xValue - (newXsize / 2));
            double newXend = Math.Round(xValue + (newXsize / 2));
            double newYstart = Math.Round(yValue - (newYsize / 2), 1);
            double newYend = Math.Round(yValue + (newYsize / 2), 1);

            area.AxisX.ScaleView.Zoom(newXstart, newXend);
            area.AxisY.ScaleView.Zoom(newYstart, newYend);
        }

        private void miOpen_Click(object sender, EventArgs e)
        {
            OpenFileDialog of = new OpenFileDialog();
            DialogResult dr = of.ShowDialog();
            if (dr == DialogResult.OK)
            {
                LoadPointsFromFile(of.FileName);
            }
        }

        void LoadPointsFromFile(string fileName)
        {
            StreamReader sr = new StreamReader(fileName);
            chart1.Series[0].Points.Clear();
            while (sr.EndOfStream == false)
            {
                string line = sr.ReadLine();
                if (line.StartsWith("#"))
                    continue;
                List<object> values = GetValues(line);
                chart1.Series[0].Points.AddXY(values[0], values[1]);
            }
        }

        private List<object> GetValues(string line)
        {
            List<object> results = new List<object>();
            results.Add(GetValueInt(ref line));
            results.Add(GetValueDouble(ref line));
            return results;
        }

        int GetValueInt(ref string str)
        {
            string part = GetValue(ref str);
            return Convert.ToInt32(part);
        }

        double GetValueDouble(ref string str)
        {
            string val = GetValue(ref str);
            val = val.Replace("D", "E");
            return Convert.ToDouble(val, new CultureInfo("en-US"));
        }

        string SkipSpace(string str)
        {
            // Everything other than numeric characters and the dot
            return Regex.Replace(str, @"\A[^0-9.\-\+DE]+", "");
        }

        string GetValue(ref string str)
        {
            str = SkipSpace(str);
            string val = Regex.Match(str, @"\A[0-9.\-\+DE]+", RegexOptions.None).ToString();
            str = str.Remove(0, val.Length);
            return val;
        }

        private void chart1_DragDrop(object sender, DragEventArgs e)
        {
            string[] FileList = (string[])e.Data.GetData(DataFormats.FileDrop, false);
            LoadPointsFromFile(FileList[0]);
        }

        private void chart1_DragEnter(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
                e.Effect = DragDropEffects.All;
            else
                e.Effect = DragDropEffects.None;
        }

        private void exitToolStripMenuItem1_Click(object sender, EventArgs e)
        {
            Close();
        }

        private void Form1_FormClosed(object sender, FormClosedEventArgs e)
        {
            Properties.Settings.Default.WindowMaximized = (WindowState == FormWindowState.Maximized);
            Properties.Settings.Default.Save();
        }

        protected override bool ProcessCmdKey(ref Message msg, Keys keyData)
        {
            if (keyData == Keys.Escape) this.Close();
            return base.ProcessCmdKey(ref msg, keyData);
        }
    }
}
