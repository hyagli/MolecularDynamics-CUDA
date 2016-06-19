using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.IO;
using System.Globalization;

namespace OutParser
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void textBox1_DragDrop(object sender, DragEventArgs e)
        {
            string[] FileList = (string[])e.Data.GetData(DataFormats.FileDrop, false);
            tbDosya.Text = FileList[0];
            tbEpot.Text = "";

            ProcessDirs(tbDosya.Text);
        }

        void ProcessDirs(string dirName)
        {
            string[] dirs = Directory.GetDirectories(dirName);
            foreach (string dir in dirs)
            {
                ProcessFile(dir + "\\mdse.out");
            }
            ProcessFile(dirName + "\\mdse.out");
        }

        void ProcessFile(string fileName)
        {
            FileStream fs;
            try
            {
                fs = new FileStream(fileName, FileMode.Open);
                byte[] array = new byte[1000];
                Encoding enc = new ASCIIEncoding();

                /*fs.Read(array, 0, 600);
                string str = enc.GetString(array);
                int pos = str.IndexOf("PP(I)") + 40;
                string val = str.Substring(pos, 8);
                tbPPZ.Text = val;
                tbPPZextra.Text = str.Substring(pos - 20, 40);
                */
                fs.Position = fs.Length - 300;
                fs.Read(array, 0, 200);

                string str = enc.GetString(array);
                int pos = str.IndexOf("EPOT=") + 5;
                string val = str.Substring(pos, 13);
                double dval = Convert.ToDouble(val, new CultureInfo("en-US"));
                val = dval.ToString();
                tbEpot.Text += val + System.Environment.NewLine;

                //Clipboard.SetText(tbEpot.Text);
                //toolStripStatusLabel1.Text = tbEpot.Text + " panoya kopyalandı.";
                fs.Close();
            }
            catch (Exception ex)
            {
                toolStripStatusLabel1.Text = "Error: " + ex.ToString();
                return;
            }
        }

        private void textBox1_DragEnter(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
                e.Effect = DragDropEffects.All;
            else
                e.Effect = DragDropEffects.None;
        }

        private void button1_Click(object sender, EventArgs e)
        {
            Form2 f = new Form2();
            f.Show();
        }

        private void aboutToolStripMenuItem_Click(object sender, EventArgs e)
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine("Opens mdse.out and parses the values for:");
            sb.AppendLine("Z axis of the periodic boundary (Third value of 'PP(I):' line at the beginning of the file.");
            sb.AppendLine("The final value of Potential Energy ('EPOT=' value at the end of the file.");
            sb.AppendLine("If the parse is unsuccessful, you can change searched positions.");
            MessageBox.Show(sb.ToString());
        }

        protected override bool ProcessCmdKey(ref Message msg, Keys keyData)
        {
            if (keyData == Keys.Escape) this.Close();
            return base.ProcessCmdKey(ref msg, keyData);
        }

        private void button2_Click(object sender, EventArgs e)
        {

        }

        private void Form1_Load(object sender, EventArgs e)
        {
            formOut2xyz form = new formOut2xyz();
            form.Show();
            Close();
        }

    }
}
