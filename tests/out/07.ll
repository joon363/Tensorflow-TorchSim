; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

define { ptr, ptr, i64, [1 x i64], [1 x i64] } @main(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, ptr %5, ptr %6, i64 %7, i64 %8, i64 %9) {
  %11 = call ptr @malloc(i64 72)
  %12 = ptrtoint ptr %11 to i64
  %13 = add i64 %12, 63
  %14 = urem i64 %13, 64
  %15 = sub i64 %13, %14
  %16 = inttoptr i64 %15 to ptr
  %17 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %11, 0
  %18 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %17, ptr %16, 1
  %19 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %18, i64 0, 2
  %20 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, i64 2, 3, 0
  %21 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, i64 1, 4, 0
  br label %22

22:                                               ; preds = %25, %10
  %23 = phi i64 [ %36, %25 ], [ 0, %10 ]
  %24 = icmp slt i64 %23, 2
  br i1 %24, label %25, label %37

25:                                               ; preds = %22
  %26 = getelementptr float, ptr %1, i64 %2
  %27 = mul nuw nsw i64 %23, %4
  %28 = getelementptr inbounds nuw float, ptr %26, i64 %27
  %29 = load float, ptr %28, align 4
  %30 = getelementptr float, ptr %6, i64 %7
  %31 = mul nuw nsw i64 %23, %9
  %32 = getelementptr inbounds nuw float, ptr %30, i64 %31
  %33 = load float, ptr %32, align 4
  %34 = fadd float %29, %33
  %35 = getelementptr inbounds nuw float, ptr %16, i64 %23
  store float %34, ptr %35, align 4
  %36 = add i64 %23, 1
  br label %22

37:                                               ; preds = %22
  ret { ptr, ptr, i64, [1 x i64], [1 x i64] } %21
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
