declare module 'cdt2d' {
  interface CDT2DOptions {
    delaunay?: boolean;
    interior?: boolean;
    exterior?: boolean;
    infinity?: boolean;
  }

  export default function cdt2d(
    points: ArrayLike<[number, number]>,
    edges?: ArrayLike<[number, number]>,
    options?: CDT2DOptions,
  ): number[][];
}
