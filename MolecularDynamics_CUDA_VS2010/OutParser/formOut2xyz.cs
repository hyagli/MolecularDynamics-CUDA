using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security;
using System.Text;
using System.Text.RegularExpressions;
using System.Windows.Forms;

namespace OutParser
{
    public partial class formOut2xyz : Form
    {
        FolderBrowserDialog FFolderBrowser;
        string xyzPath;
        private string basePath;

        public formOut2xyz()
        {
            InitializeComponent();
            FFolderBrowser = new FolderBrowserDialog();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            FFolderBrowser.SelectedPath = tbPath.Text;
            DialogResult result = FFolderBrowser.ShowDialog();
            if (result == System.Windows.Forms.DialogResult.OK)
            {
                tbPath.Text = FFolderBrowser.SelectedPath;
                BuildFiles(tbPath.Text, false);
            }
        }

        private void BuildFiles(string path, bool process)
        {
            string[] files = Directory.GetFiles(path, "*.out");
            if (files.Count() > 0)
            {
                tbLog.AppendText(files[0] + "\n");
                if (process)
                    ProcessFile(files[0]);
            }

            string[] folders = Directory.GetDirectories(path);
            List<string> list = folders.ToList();
            list.Sort(new NaturalStringComparer());
            foreach (var item in list)
            {
                BuildFiles(item, process);
            }
        }

        private void ProcessFile(string path)
        {
            StreamWriter out1 = null;
            StreamWriter outScript = null;
            int writeAfter = 0;
            int writeNow = 0;
            string numAtoms = "";
            string fileName = "";
            string mdsNo = "";

            string folderName = path.Replace(basePath, "");
            folderName = folderName.Replace("mdse.out", "");
            folderName = folderName.Replace("\\", "");

            outScript = File.AppendText(xyzPath + "JMol_script.txt");

            Regex rgx = new Regex(@"\S+");


            string[] lines = File.ReadAllLines(path);
            foreach (var line in lines)
            {
                if (line.Contains("NUMBER OF MOVING ATOMS"))
                {
                    numAtoms = Regex.Match(line, @"(\d+)$").Value;
                    continue;
                }

                if (line.Contains("PERIODIC PRINTING OF COORDINATES"))
                {
                    mdsNo = Regex.Match(line, @"(\d+)$").Value;
                    
                    if (folderName != "")
                        fileName = string.Format("{0}_{1}.xyz", folderName, mdsNo.PadLeft(9, '0'));
                    else
                        fileName = string.Format("stepLast_{0}.xyz", mdsNo.PadLeft(9, '0'));
                    out1 = File.CreateText(xyzPath + fileName);

                    outScript.WriteLine("load " + xyzPath + fileName);
                    outScript.WriteLine("rotate 90");
                    outScript.WriteLine("write " + xyzPath + fileName + ".jpg");

                    writeAfter = 5;
                    continue;
                }

                if (writeAfter > 0)
                {
                    writeAfter--;
                    if (writeAfter == 0)
                    {
                        writeNow = 2;
                        continue;
                    }
                }

                if (writeNow > 0)
                {
                    if (line == "")
                    {
                        writeNow = 0;
                        out1.Close();
                        continue;
                    }

                    string outLine = "Cu";
                    MatchCollection sayilar = rgx.Matches(line);
                    for (int i = 1; i < 4; i++)
                        outLine += " " + sayilar[i].Value;

                    if (writeNow == 2)
                    {
                        out1.WriteLine(numAtoms);
                        out1.WriteLine(fileName);
                        writeNow--;
                    }
                    out1.WriteLine(outLine);
                }
            }

            outScript.Close();
        }


        public sealed class NaturalStringComparer : IComparer<string>
        {
            private readonly int modifier = 1;

            public NaturalStringComparer(bool descending)
            {
                if (descending)
                    modifier = -1;
            }

            public NaturalStringComparer()
                : this(false) { }

            public int Compare(string a, string b)
            {
                return SafeNativeMethods.StrCmpLogicalW(a ?? "", b ?? "") * modifier;
            }
        }

        public sealed class NaturalFileInfoComparer : IComparer<FileInfo>
        {
            public int Compare(FileInfo a, FileInfo b)
            {
                return SafeNativeMethods.StrCmpLogicalW(a.Name ?? "", b.Name ?? "");
            }
        }

        [SuppressUnmanagedCodeSecurity]
        internal static class SafeNativeMethods
        {
            [DllImport("shlwapi.dll", CharSet = CharSet.Unicode)]
            public static extern int StrCmpLogicalW(string psz1, string psz2);
        }

        private void btBuild_Click(object sender, EventArgs e)
        {
            if (!Directory.Exists(tbPath.Text))
                return;

            basePath = tbPath.Text.TrimEnd('\\') + "\\";
            xyzPath = basePath + "xyz\\";

            if (Directory.Exists(xyzPath))
            {
                tbLog.AppendText("Directory exists: " + xyzPath + "\n");
                return;
            }

            Directory.CreateDirectory(xyzPath);

            BuildFiles(tbPath.Text, true);
        }
    }
}
