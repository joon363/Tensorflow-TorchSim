	.text
	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_zicsr2p0_zmmul1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl256b1p0_zvl32b1p0_zvl64b1p0_xsfvcp1p0"
	.file	"LLVMDialectModule"
	.globl	main                            # -- Begin function main
	.p2align	1
	.type	main,@function
main:                                   # @main
.Lfunc_begin0:
	.cfi_startproc
# %bb.0:
	addi	sp, sp, -64
	.cfi_def_cfa_offset 64
	sd	ra, 56(sp)                      # 8-byte Folded Spill
	sd	s0, 48(sp)                      # 8-byte Folded Spill
	sd	s1, 40(sp)                      # 8-byte Folded Spill
	sd	s2, 32(sp)                      # 8-byte Folded Spill
	sd	s3, 24(sp)                      # 8-byte Folded Spill
	sd	s4, 16(sp)                      # 8-byte Folded Spill
	sd	s5, 8(sp)                       # 8-byte Folded Spill
	sd	s6, 0(sp)                       # 8-byte Folded Spill
	.cfi_offset ra, -8
	.cfi_offset s0, -16
	.cfi_offset s1, -24
	.cfi_offset s2, -32
	.cfi_offset s3, -40
	.cfi_offset s4, -48
	.cfi_offset s5, -56
	.cfi_offset s6, -64
	ld	s5, 80(sp)
	ld	s6, 64(sp)
	mv	s1, a7
	mv	s2, a5
	mv	s3, a3
	mv	s4, a2
	mv	s0, a0
	li	a0, 72
	call	malloc
	li	a4, 0
	addi	a1, a0, 63
	andi	a3, a1, -64
	li	a1, 1
	li	a6, 2
	slli	s6, s6, 2
	add	s1, s1, s6
	slli	a5, s5, 2
	slli	s3, s3, 2
	add	s4, s4, s3
	slli	s2, s2, 2
	mv	a2, a3
	bltz	a1, .LBB0_2
.LBB0_1:                                # =>This Inner Loop Header: Depth=1
	flw	fa5, 0(s4)
	flw	fa4, 0(s1)
	fadd.s	fa5, fa5, fa4
	fsw	fa5, 0(a2)
	addi	a4, a4, 1
	addi	a2, a2, 4
	add	s1, s1, a5
	add	s4, s4, s2
	bge	a1, a4, .LBB0_1
.LBB0_2:
	sd	zero, 16(s0)
	sd	a0, 0(s0)
	sd	a3, 8(s0)
	sd	a6, 24(s0)
	sd	a1, 32(s0)
	ld	ra, 56(sp)                      # 8-byte Folded Reload
	ld	s0, 48(sp)                      # 8-byte Folded Reload
	ld	s1, 40(sp)                      # 8-byte Folded Reload
	ld	s2, 32(sp)                      # 8-byte Folded Reload
	ld	s3, 24(sp)                      # 8-byte Folded Reload
	ld	s4, 16(sp)                      # 8-byte Folded Reload
	ld	s5, 8(sp)                       # 8-byte Folded Reload
	ld	s6, 0(sp)                       # 8-byte Folded Reload
	addi	sp, sp, 64
	ret
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
	.section	.stack_sizes,"o",@progbits,.text
	.quad	.Lfunc_begin0
	.byte	64
	.text
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
