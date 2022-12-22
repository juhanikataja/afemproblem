//+
SetFactory("OpenCASCADE");
Box(1) = {0, 0, 0, 1, 1, 1};
//+
Field[1] = MathEval;
//+
Field[1].F = "0.2";
//+
Background Field = 1;
//+
Field[1].F = "0.5";
//+
Field[1].F = "1";
