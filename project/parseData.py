import lyricsgenius
import csv

genius = lyricsgenius.Genius("XWWEtd9o3o8Ul5MfqyEfksX0v5NZ59WdDstvcPeglauUxGJ8wre61z_wSe-Jt26I")
#artist = genius.search_artist("Bruno Mars", max_songs=1, sort="title")

#song = genius.search_song("Blinding Lights", "The Weeknd")
#print(song.lyrics)

with open("top10s.csv", newline='') as file:
    reader = csv.reader(file)
    
    with open("top10s_with_lyrics.csv", "w", newline='') as outputFile:
        writer = csv.writer(outputFile)

        for row in reader:
            title = row[1]
            artist = row[2]
            genre = row[3]
            key = str(title + " " + artist)
            
            try:
                writer.writerow([title, artist, genius.search_song(title, artist).lyrics, genre])
            except:
                pass
            #print("\n===== Row Written to CSV =====")
            #print(songs)
            #print("===== END OF SONGS ====='n")
            #print("Song Title: " + row[1] + " Artist Name: " + row[2] + " Genre: " + row[3])
