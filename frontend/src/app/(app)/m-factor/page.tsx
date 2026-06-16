import { connection } from "next/server";
import { Newsreader, IBM_Plex_Sans, IBM_Plex_Mono } from "next/font/google";
import MFactorClient from "./m-factor-client";

const newsreader = Newsreader({ variable: "--font-newsreader", subsets: ["latin"], weight: ["400","500","600","700"], style: ["normal","italic"] });
const plexSans = IBM_Plex_Sans({ variable: "--font-plex-sans", subsets: ["latin"], weight: ["400","500","600"] });
const plexMono = IBM_Plex_Mono({ variable: "--font-plex-mono", subsets: ["latin"], weight: ["400","500","600"] });

export default async function MFactorPage() {
  await connection();
  return (
    <div className={`contents ${newsreader.variable} ${plexSans.variable} ${plexMono.variable}`}>
      <MFactorClient />
    </div>
  );
}
